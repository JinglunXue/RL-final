import numpy as np
import random
import torch

class Retail_Environment(object):
    def __init__(self, lifetime, leadtime, mean_demand, coef_of_var, max_order, cost_order, cost_outdate, cost_lost,
                 cost_holding, FIFO, LIFO, time):
        '''
        initialize the Environment of agent.
        Parameters corresponds to the notations in the paper:
            lifetime:m
            leadtime:L
            demand~gamma distribution(mean_demand,coef_of_var)
            action space should be limited：0,1,...,max_order
            cost_order,cost_outdate,cost_lost,cost_holding : cost parameters used to calculate reward.
            FIFO,LIFO : binary variables, indicate First In First Out or Last In First Out
            time:how many episodes(1000)
        '''
        self.lifetime = lifetime
        self.leadtime = leadtime
        self.mean_demand = mean_demand
        self.coef_of_var = coef_of_var
        self.max_order = max_order
        self.cost_order = cost_order
        self.cost_outdate = cost_outdate
        self.cost_lost = cost_lost
        self.cost_holding = cost_holding
        self.FIFO = FIFO
        self.LIFO = LIFO
        self.time = time
        self.demand = 0
        self.action = 0
        self.current_time = 0
        self.reward = 0
        self.shape = 1 / (self.coef_of_var ** 2)
        self.scale = self.mean_demand / self.shape
        #dimension of state space：m+L-1
        self.state = []
        for i in range(self.lifetime + self.leadtime - 1):
            self.state.append(0)

        self.render_state = self.state.copy()
        self.action_space = []
        for i in range(self.max_order + 1):
            self.action_space.append(i)

        print('Environment created...')

    def step(self, action):
        '''when an action is taken，how should state change，what reward should we get，is this the last episode'''
        if type(action)==int:
            self.action = action
        else:
            self.action = action.argmax()
        self.demand = round(np.random.gamma(self.shape, self.scale, size=None))#每一个episode读取一个demand
        demand = self.demand
        self.render_state = self.state.copy()

        # update inventory in pipeline with order
        next_state = [None] * (self.leadtime + self.lifetime)
        next_state[0] = self.action
        for i in range(self.leadtime + self.lifetime - 1):
            next_state[i + 1] = self.state[i]

        # inventory depletion
        calc_state = next_state.copy()
        if self.FIFO:
            for i in range(self.lifetime):
                if demand > 0:
                    next_state[-i - 1] = max(calc_state[-i - 1] - demand, 0)
                    demand = max(demand - calc_state[-i - 1], 0)
        if self.LIFO:
            for i in range(self.leadtime, self.leadtime + self.lifetime):
                if demand > 0:
                    next_state[i] = max(calc_state[i] - demand, 0)
                    demand = max(demand - calc_state[i], 0)

        # age inventory
        calc_state = next_state.copy()
        for i in range(self.leadtime + self.lifetime):
            if i == 0:
                next_state[i] = 0
            else:
                next_state[i] = calc_state[i - 1]
        #calculate reward
        order_cost = self.action * self.cost_order
        outdate_cost = calc_state[-1] * self.cost_outdate
        lost_sales_cost = demand * self.cost_lost
        holding_cost = 0
        for i in range(self.leadtime + 1, self.leadtime + self.lifetime):
            holding_cost += next_state[i] * self.cost_holding

        self.reward = -order_cost - outdate_cost - lost_sales_cost - holding_cost#reward should be negative
        self.current_time += 1
        for i in range(self.lifetime + self.leadtime - 1):
            self.state[i] = next_state[i + 1]

        return self.state, self.reward, self.isFinished(self.current_time)

    def isFinished(self, current_time):
        '''is cuurent_time the last episode'''
        return current_time == self.time

    def reset(self):
        '''restart a period'''
        self.current_time = 0
        self.state = []
        for i in range(self.leadtime + self.lifetime - 1):
            self.state.append(0)
        # print('Reset environment...')
        return self.state, self.current_time

    def render(self):
        '''print current state'''
        print('---------------------------------------------------')
        print('*****   Period ' + str(self.current_time) + '   *****')
        inventory_on_hand = []
        for i in range(self.leadtime - 1, self.leadtime + self.lifetime - 1):
            inventory_on_hand.append(self.render_state[i])
        inventory_in_pipeline = []
        inventory_in_pipeline.append(self.action)
        for i in range(0, self.leadtime - 1):
            inventory_in_pipeline.append(self.render_state[i])
        print('Inventory on hand: ' + str(inventory_on_hand))
        print('Order placed: ' + str(self.action))
        print('Orders in pipeline: ' + str(inventory_in_pipeline))
        print('Demand encountered: ' + str(self.demand))
        print('Costs: ' + str(self.reward))

    def random_action(self):
        '''take a action randomly'''
        return random.sample(self.action_space, 1)[0]

    def random_state(self):
        '''sample a state randomly'''
        random_state = []
        for i in range(0, self.lifetime + self.leadtime - 1):
            random_state.append(np.random.randint(self.max_order))
        return random_state