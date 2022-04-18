import numpy as np
from ENV_TRAIN import Retail_Environment
import random
import os
from pandas import DataFrame

class Agent_B():
    def __init__(self, state_size, action_size, epochs, env,
                  iteration, base_stock, x):

        self.state_size = state_size
        self.action_size = action_size

        self.epochs = epochs
        self.env = env
        self.epoch_counter = 0
        self.iteration = iteration
        self.x = x

        self.base_stock = base_stock

    def play(self):
        scores = []
        for e in range(self.epochs):
            done = False
            score = 0
            state, _ = self.env.reset()
            while not done:
                state = np.reshape(state, [1, self.state_size])
                in_inv = 0
                for i in range(self.env.leadtime + self.env.lifetime - 1):
                    in_inv += state[0][i]
                bsp_order = max(0, self.base_stock - in_inv)

                next_state, reward, done= self.env.step(bsp_order)
                score += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state

            avg_score = score / self.env.time
            self.epoch_counter += 1

            print('Epoch ' + str(self.epoch_counter) + ' | Avg score per period: ' + str(-avg_score))
            scores.append(-avg_score)

        '''df = DataFrame({'Reward': scores})
        path = PATH
        df.to_excel(str(path) + str(self.x) + '/Lifetime ' + str(self.env.lifetime) + ' - iteration ' + str(
            self.iteration) + '.xlsx')
        '''
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
#################################parameters about reward shaping
FACTOR = 50
BASESTOCK = 0

env_train = Retail_Environment(LIFETIME, LEADTIME, MEAN_DEMAND, CV, MAX_ORDER, C_ORDER, C_PERISH, C_LOST, C_HOLD, FIFO,
                               LIFO, TRAIN_TIME)
state_size = len(env_train.state)
action_size = len(env_train.action_space)

rows = 50
columns = EPOCHS
scores_B = [[0 for i in range(columns)] for j in range(rows)]
for i in range(0, 10):
    agent_B=Agent_B(state_size, action_size, EPOCHS, env_train, 0, BASESTOCK, x)
    scores_B[i]=agent_B.play()
dict={}
for i in range(10):
    dict[str(i)] = scores_B[i]
df_B = DataFrame(dict)
path = os.getcwd()
df_B.to_excel(path + '/EVAL' + str(x) + '/overview_B.xlsx')

