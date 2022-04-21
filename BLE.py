import numpy as np
from ENV_TRAIN import Retail_Environment
import random
import os
from pandas import DataFrame

class Agent_BLE():
    '''agent who use BLE policy '''
    def __init__(self, state_size, action_size, epochs, env,
                  iteration, S1, S2,b, x):

        self.state_size = state_size
        self.action_size = action_size

        self.epochs = epochs
        self.env = env
        self.epoch_counter = 0
        self.iteration = iteration
        self.x = x

        self.S1 = S1
        self.S2 = S2
        self.b = b
        self.alpha = (1 - ((S2 - S1) / b))

    def play(self):
        '''return a list，recording the average cost in every epoch of 2000 epochs'''
        scores = []
        for e in range(self.epochs):
            done = False
            score = 0
            state, _ = self.env.reset()
            while not done:
                state = np.reshape(state, [1, self.state_size])
                EW = 0
                demand_left = 0
                adj_state = []
                for i in range(self.env.leadtime):
                    if self.env.FIFO == True:
                        EW += max(0, self.env.state[-1 - i] - self.env.mean_demand - demand_left)
                        demand_left = max(0, (demand_left + self.env.mean_demand) - self.env.state[-1 - i])
                    elif self.env.LIFO == True:
                        dem = self.env.mean_demand
                        k = self.env.leadtime - 1
                        j = 0
                        while dem > 0 and j <= self.env.lifetime - 1:
                            in_store = adj_state[k + j]
                            adj_state[k + j] = max(0, adj_state[k + j] - dem)
                            dem = max(0, dem - in_store)
                            j += 1
                        EW += max(0, adj_state[-1])
                        for p in range(self.env.leadtime + self.env.lifetime - 2):
                            adj_state[-1 - p] = adj_state[-2 - p]
                        adj_state[0] = 0

                in_inv = 0
                for i in range(self.env.leadtime + self.env.lifetime - 1):
                    in_inv += state[0][i]

                if in_inv < self.b:
                    order = max(0, round(self.S1 - (self.alpha * in_inv) + EW))
                else:
                    order = max(0, self.S2 - in_inv + EW)

                next_state, reward, done= self.env.step(order)
                score += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state

            avg_score = score / self.env.time
            self.epoch_counter += 1

            print('Epoch ' + str(self.epoch_counter) + ' | Avg score per period: ' + str(-avg_score))
            scores.append(-avg_score)

        return scores

x = 0
#################################
MEAN_DEMAND = 4
CV = 0.5
LIFETIME = 2
LEADTIME = 1
C_LOST = 5
C_HOLD = 1
C_PERISH = 7
C_ORDER = 3
FIFO = True#False
LIFO = False#True
#################################
MAX_ORDER = 10
TRAIN_TIME = 1000
#################################
GAMMA = 0.99
EPSILON_DECAY = 0.997
PSI_DECAY = 0.99
EPSILON_MIN = 0.01
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 32
UPDATE = 20
#################################
FACTOR = 50
S1 = 0
S2 = 0
b = 0.1#0
env_train = Retail_Environment(LIFETIME, LEADTIME, MEAN_DEMAND, CV, MAX_ORDER, C_ORDER, C_PERISH, C_LOST, C_HOLD, FIFO,
                               LIFO, TRAIN_TIME)
state_size = len(env_train.state)
action_size = len(env_train.action_space)

rows = 50
columns = EPOCHS
scores_BLE = [[0 for i in range(columns)] for j in range(rows)]
for i in range(0, 10):
    agent_BLE=Agent_BLE(state_size, action_size, EPOCHS, env_train, 0, S1, S2,b, x)
    scores_BLE[i]=agent_BLE.play()
dict={}
for i in range(10):
    dict[str(i)] = scores_BLE[i]
df_BLE = DataFrame(dict)
path = os.getcwd()
df_BLE.to_excel(path + '/EVAL' + str(x) + '/overview_BLE.xlsx')