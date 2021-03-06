import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from pandas import DataFrame

class Net(nn.Module):
    '''module for network of DQN'''
    def __init__(self,state_size,action_size):
        super(Net,self).__init__()
        self.fc1=nn.Linear(state_size,32)
        #self.fc1.weight.data.normal_(0, 0.1)
        self.fc2=nn.Linear(32,32)
        #self.fc2.weight.data.normal_(0, 0.1)
        self.fc3=nn.Linear(32,action_size)
        #self.fc3.weight.data.normal_(0, 0.1)
    def forward(self,x):
        '''input state, output action value choose by DQN agent'''
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        action_value=self.fc3(x)
        return action_value

class DQN_Agent_BLE():

    def __init__(self, state_size, action_size, gamma, epsilon_decay, epsilon_min, learning_rate, epochs, env, batch_size,
                 update, iteration, S1, S2, b, factor, x):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=20000)
        self.gamma = gamma  # discount factor, used to calculate value function

        # epsilon-greedy policy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

        self.epochs = epochs#one epoch consists of 2000 periods
        self.env = env
        self.batch_size = batch_size
        self.update = update
        self.epoch_counter = 0
        self.epsilon = 1.0
        self.iteration = iteration#10 agents will be trained, which one is at present
        self.x = x

        self.model = Net(self.state_size, self.action_size)
        self.target_model = Net(self.state_size, self.action_size)

        self.S1 = S1
        self.S2 = S2
        self.b = b
        self.alpha = (1 - ((S2 - S1) / b))
        self.factor = factor

        # define optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func=nn.MSELoss()
        #self.loss_func = nn.L1Loss()

    def act(self, state):
        '''according to state,return the action index we should choose'''
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon:
            action_index = random.randrange(self.action_size)
            action[action_index] = 1
        else:
            act_values = self.model.forward(torch.flatten(torch.from_numpy(state).float()))
            action_index = torch.argmax(act_values)
            action[action_index] = 1
        return action  # array

    def remember(self, state, action, reward, next_state, done):
        '''put the sequence into replay memory'''
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        '''called every epoch???1000 episodes)'''
        # experience replay from replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = torch.stack([torch.flatten(torch.from_numpy(data[0]).float()) for data in minibatch])  # .cuda(2)
        action_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).float()  # .cuda(2)
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])).float()  # .cuda(2)
        nextState_batch = torch.stack(
            [torch.flatten(torch.from_numpy(data[3]).float()) for data in minibatch])  # .cuda(2)

        QValue_batch = self.model.forward(state_batch)
        Q_Action = torch.sum(torch.mul(action_batch, QValue_batch), dim=1).float()
        QTValue_batch = self.target_model.forward(nextState_batch)
        y_batch = reward_batch + torch.from_numpy((1 - np.array(minibatch)[:, -1]).astype(float)) * self.gamma * \
                  torch.max(QTValue_batch, 1)[0]  # .cuda(2)\
        y_batch = y_batch.to(torch.float32).detach()

        loss = self.loss_func(Q_Action, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update target network
        if self.epoch_counter % self.update == 0:
            self.update_target_model()

    def update_target_model(self):
        '''update the target model using parameters of self.model'''
        self.target_model.fc1.weight.data = self.model.fc1.weight.data.clone()
        self.target_model.fc1.bias.data = self.model.fc1.bias.data.clone()
        self.target_model.fc2.weight.data = self.model.fc2.weight.data.clone()
        self.target_model.fc2.bias.data = self.model.fc2.bias.data.clone()
        self.target_model.fc3.weight.data = self.model.fc3.weight.data.clone()
        self.target_model.fc3.bias.data = self.model.fc3.bias.data.clone()
        print('***** Target network updated *****')

    def train(self):
        '''return a list???recording the average cost in every epoch of 2000 epochs'''
        scores = []
        for e in range(self.epochs):
            done = False
            score = 0
            state, _ = self.env.reset()
            prev_val = 0
            while not done:
                state = np.reshape(state, [1, self.state_size])
                action = self.act(state)
                next_state, reward, done= self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                ########################calculate F
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
                cur_val = -self.factor * abs(order - action.argmax())
                F = cur_val - ((1 / self.gamma) * prev_val)
                #######################
                prev_val = cur_val
                total = reward + F # the shaped reward
                self.remember(state, action, total, next_state, done)
                state = next_state

            avg_score = score / self.env.time
            self.epoch_counter += 1

            print('Epoch ' + str(self.epoch_counter) + ' | Avg score per period: ' + str(-avg_score))

            if len(self.memory) > self.batch_size:
                self.replay()

            scores.append(-avg_score)

        path = os.getcwd()
        #save model
        self.save(path + str(self.x) + '/Lifetime ' + str(self.env.lifetime) + ' - iteration ' + str(self.iteration)+'.pth')
        return scores

    def save(self,name):
        '''save model'''
        torch.save(self.model, name)

    def load(self,name):
        '''load model'''
        torch.load(self.model, name)