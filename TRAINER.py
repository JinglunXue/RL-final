from ENV_TRAIN import Retail_Environment
from UNSHAPED_DQN_pytorch import DQN_Agent
from SHAPED_BLE_pytorch import DQN_Agent_BLE
from SHAPED_B_pytorch import DQN_Agent_B
from pandas import DataFrame
import os

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
S1 = 0
S2 = 0
b = 0.1#0

#create a environment for agents
env_train = Retail_Environment(LIFETIME, LEADTIME, MEAN_DEMAND, CV, MAX_ORDER, C_ORDER, C_PERISH, C_LOST, C_HOLD, FIFO,
                               LIFO, TRAIN_TIME)
state_size = len(env_train.state)
action_size = len(env_train.action_space)

rows = 50
columns = EPOCHS
scores_DQN_Agent = [[0 for i in range(columns)] for j in range(rows)]
scores_DQN_Agent_B = [[0 for i in range(columns)] for j in range(rows)]
scores_DQN_Agent_BLE = [[0 for i in range(columns)] for j in range(rows)]

#10 agents are trained
for i in range(0, 10):
    print("New agent created...")
    # create unshaped DQN agent
    DQN_agent = DQN_Agent(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env_train,BATCH_SIZE, UPDATE, i, 1)
    # create shaped_b agent
    DQN_agent_B = DQN_Agent_B(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env_train, BATCH_SIZE, UPDATE, i, BASESTOCK, FACTOR, 2)
    # create shaped_ble agent
    DQN_agent_BLE = DQN_Agent_BLE(state_size, action_size, GAMMA, EPSILON_DECAY, EPSILON_MIN, LEARNING_RATE, EPOCHS, env_train, BATCH_SIZE, UPDATE, i, S1, S2, b, FACTOR, 3)

    scores_DQN_Agent[i] = DQN_agent.train()
    scores_DQN_Agent_B[i] = DQN_agent_B.train()
    scores_DQN_Agent_BLE[i] = DQN_agent_BLE.train()
dict1={}
dict2={}
dict3={}
for i in range(10):
    dict1[str(i)] = scores_DQN_Agent[i]
    dict2[str(i)] = scores_DQN_Agent_B[i]
    dict3[str(i)] = scores_DQN_Agent_BLE[i]
df_DQN_Agent = DataFrame(dict1)
df_DQN_Agent_B = DataFrame(dict2)
df_DQN_Agent_BLE = DataFrame(dict3)
#write results into excel
path = os.getcwd()
df_DQN_Agent.to_excel(path + '/EVAL' + str(x) + '/overview.xlsx')
df_DQN_Agent_B.to_excel(path + '/EVAL' + str(x) + '/overview_DQN_B.xlsx')
df_DQN_Agent_BLE.to_excel(path + '/EVAL' + str(x) + '/overview_DQN_BLE.xlsx')