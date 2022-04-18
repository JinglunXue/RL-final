import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from pandas import DataFrame

class Net(nn.Module):
    def __init__(self,state_size,action_size):
        super(Net,self).__init__()
        self.fc1=nn.Linear(state_size,32)
        #self.fc1.weight.data.normal_(0, 0.1)
        self.fc2=nn.Linear(32,32)
        #self.fc2.weight.data.normal_(0, 0.1)
        self.fc3=nn.Linear(32,action_size)
        #self.fc3.weight.data.normal_(0, 0.1)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        action_value=self.fc3(x)
        return action_value

class DQN_Agent():
    def __init__(self, state_size, action_size, gamma, epsilon_decay, epsilon_min, learning_rate, epochs, env, batch_size, update, iteration, x):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.memory = deque(maxlen=20000)
        self.batch_size = batch_size

        self.gamma = gamma#discount factor, used to calculate value function
        #epsilon-greedy policy
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.epoch_counter = 0
        self.iteration = iteration#10 agents will be trained, which one is at present
        self.epochs = epochs#one epoch consists of 2000 periods

        self.env = env#ENV_TRAIN

        self.update = update#target network is updated every 'epochs' epochs

        self.x = x

        self.model = Net(self.state_size,self.action_size)
        self.target_model = Net(self.state_size,self.action_size)

        #define optimizer
        self.learning_rate=learning_rate
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.loss_func=nn.MSELoss()
        # self.loss_func = nn.L1Loss()

    def act(self,state):
        '''according to state,return the action index we should choose'''
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon:
            action_index=random.randrange(self.action_size)
            action[action_index] = 1
        # print(state)
        else:
            act_values = self.model.forward(torch.flatten(torch.from_numpy(state).float()))
            action_index = torch.argmax(act_values)
            action[action_index] = 1
        return action#array, like [1,0,0,0,0]

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state, action, reward, next_state, done))#(state, action, reward, next_state, done) is one experience

    def replay(self):
        '''called every epoch（1000 episodes)'''
        # experience replay from replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch=torch.stack([torch.flatten(torch.from_numpy(data[0]).float()) for data in minibatch])
        action_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).float()
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])).float()
        nextState_batch = torch.stack([torch.flatten(torch.from_numpy(data[3]).float()) for data in minibatch])

        QValue_batch = self.model.forward(state_batch)
        Q_Action = torch.sum(torch.mul(action_batch, QValue_batch), dim=1).float()
        QTValue_batch = self.target_model.forward(nextState_batch)
        y_batch = reward_batch + torch.from_numpy((1 - np.array(minibatch)[:, -1]).astype(float))* self.gamma * torch.max(QTValue_batch, 1)[0]
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

    def train(self):
        '''return a list，recording the average cost in every epoch of 2000 epochs'''
        scores = []
        for e in range(self.epochs):
            done = False
            score = 0
            state, _ = self.env.reset()#return self.state, self.current_time
            while not done:#1000periods
                state = np.reshape(state, [1, self.state_size])
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
            avg_score = score / self.env.time
            self.epoch_counter += 1

            print('Epoch ' + str(self.epoch_counter) + ' | Avg score per period: ' + str(-avg_score))

            if len(self.memory) > self.batch_size:
                self.replay()

            scores.append(-avg_score)#avg_score is negative

        path = os.getcwd()
        #save model
        self.save(path + str(self.x) + '/Lifetime ' + str(self.env.lifetime) + ' - iteration ' + str(self.iteration)+'.pth')

        return scores

    def update_target_model(self):
        self.target_model.fc1.weight.data = self.model.fc1.weight.data.clone()
        self.target_model.fc1.bias.data = self.model.fc1.bias.data.clone()
        self.target_model.fc2.weight.data = self.model.fc2.weight.data.clone()
        self.target_model.fc2.bias.data = self.model.fc2.bias.data.clone()
        self.target_model.fc3.weight.data = self.model.fc3.weight.data.clone()
        self.target_model.fc3.bias.data = self.model.fc3.bias.data.clone()
        print('***** Target network updated *****')

    def save(self,name):
        torch.save(self.model, name)

    def load(self,name):
        torch.load(self.model, name)

