"""A trading environment"""
import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from utils import get_sim_path, get_sim_path_sabr


class TradingEnv(gym.Env):
    """
    trading environment;
    contains observation space (already simulated samples), action space, reset(), step()
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day; time to maturity in unit of day
    def __init__(self, cash_flow_flag=0, dg_random_seed=1, num_sim=500002, sabr_flag = False,
        continuous_action_flag=False, spread=0, init_ttm=5, trade_freq=1, num_contract=1):
        
        # observation space 
        # simulated data: array of asset price, option price and delta paths (num_path x num_period)
        # generate data now
        if sabr_flag:
            self.path, self.option_price_path, self.delta_path, self.bartlett_delta_path = get_sim_path_sabr(M=init_ttm, freq=trade_freq,
                np_seed=dg_random_seed, num_sim=num_sim)
        else:
            self.path, self.option_price_path, self.delta_path = get_sim_path(M=init_ttm, freq=trade_freq,
                np_seed=dg_random_seed, num_sim=num_sim)

        # other attributes
        self.num_path = self.path.shape[0]

        # set num_period: initial time to maturity * daily trading freq + 1 (see get_sim_path() in utils.py): (s0,s1...sT) -->T+1
        self.num_period = self.path.shape[1]
        # print("***", self.num_period)

        # time to maturity array
        self.ttm_array = np.arange(init_ttm, -trade_freq, -trade_freq)
        # print(self.ttm_array)

        # spread
        self.spread = spread                                                                                   ## tick???spread cost???

        # step function initialization depending on cash_flow_flag
        if cash_flow_flag == 1:
            self.step = self.step_cash_flow   # see step_cash_flow() definition below. Internal reference use self.
        else:
            self.step = self.step_profit_loss

        self.num_contract = num_contract
        self.strike_price = 100

        # track the index of simulated path in use
        self.sim_episode = -1

        # track time step within an episode (it's step)
        self.t = None

        # action space                                                                                                        #action space justify?
        if continuous_action_flag:
            self.action_space = spaces.Box(low=np.array([0]), high=np.array([num_contract * 100]), dtype=np.float32)
        else:
            self.num_action = num_contract * 100 + 1
            self.action_space = spaces.Discrete(self.num_action)  #number from 0 to self.num_action-1

        self.num_state = 3

        self.state = []    # initialize current state

        # seed and start
        self.seed()  # call this function when intialize ...
        # self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)  #self.np_random now is a generateor np.random.RandomState() with a strong random seed; seed is a strong random seed
        return [seed]

    def reset(self):
        # repeatedly go through available simulated paths (if needed)
        # go to the starting point again
        self.sim_episode = (self.sim_episode + 1) % self.num_path

        self.t = 0

        price = self.path[self.sim_episode, self.t]
        position = 0

        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        return self.state

    def step_cash_flow(self, action):
        """
        cash flow period reward
        take a step and return self.state, reward, done, info
        """

        # do it consistently as in the profit & loss case
        # current prices (at t)
        current_price = self.state[0]

        # current position
        current_position = self.state[1]

        # update time/period
        self.t = self.t + 1

        # get state for tomorrow
        price = self.path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]   #state transist to next price, ttm and stores current action(position)

        # calculate period reward (part 1)
        cash_flow = -(position - current_position) * current_price - np.abs(position - current_position) * current_price * self.spread    # change cost formula????

        # if tomorrow is end of episode, only when at the end day done=True , self.num_period = T/frequency +1
        if self.t == self.num_period - 1:
            done = True   #you have arrived at the terminal
            # add (stock payoff + option payoff) to cash flow
            reward = cash_flow + price * position - max(price - self.strike_price, 0) * self.num_contract * 100 - position * price * self.spread   # change cost formula????
        else:
            done = False
            reward = cash_flow

        # for other info
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info

    def step_profit_loss(self, action):
        """
        profit loss period reward
        """

        # current prices (at t)
        current_price = self.state[0]
        current_option_price = self.option_price_path[self.sim_episode, self.t]

        # current position
        current_position = self.state[1]

        # update time
        self.t = self.t + 1

        # get state for tomorrow (at t + 1)
        price = self.path[self.sim_episode, self.t]
        option_price = self.option_price_path[self.sim_episode, self.t]
        position = action
        ttm = self.ttm_array[self.t]

        self.state = [price, position, ttm]

        # calculate period reward (part 1)
        reward = (price - current_price) * position - np.abs(current_position - position) * current_price * self.spread

        # if tomorrow is end of episode
        if self.t == self.num_period - 1:
            done = True
            reward = reward - (max(price - self.strike_price, 0) - current_option_price) * self.num_contract * 100 - position * price * self.spread  #liquidate option and stocks
        else:
            done = False
            reward = reward - (option_price - current_option_price) * self.num_contract * 100

        # for other info later
        info = {"path_row": self.sim_episode}

        return self.state, reward, done, info
