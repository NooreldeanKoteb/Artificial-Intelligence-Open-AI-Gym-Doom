# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 21:19:54 2020

@author: Nooreldean Koteb
"""


#AI for Doom

#Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Importing the packges for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

#Importing other python files
import experience_replay, image_preprocessing

#Part 1 - Building the AI
#Making the brain
class CNN(nn.Module):
    
    #Initlializing the class
    def __init__(self, number_actions):
        #Inheriting from nn.Model
        super(CNN, self).__init__()
        
        #Convolution layers
        #This will be in black and white so 1 in channel, 3 channel is colored (other layers its last output)
        #Out_channels is 32 because we want 32 images
        #kernel_size we will use a 5x5 feature detector and lower the size for more layers
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        
        #Full connection Neural Network
        #count_neurons is taking in size of images resized from doom to 80x80
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
    
    #Images dimensions from Doom will be resized to 80x80
    #Convolutional propagation, Max pooling, and Flattening to get count
    def count_neurons(self, image_dim):
        #(batch, dim, dim) (* will allow tuple to pass as a list)
        x = Variable(torch.rand(1, *image_dim))
        
        #Convolutional layers
        #Max pooling & neuron activation
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        
        #Flattening layer
        return x.data.view(1, -1).size(1)
    
    #Propagation
    def forward(self, x):
        #Convolutional propagation, Max pooling
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        
        #Flattening
        x = x.view(x.size(0), -1)
        
        #Propagating through Neural Network   
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        

#Making the body
class SoftmaxBody(nn.Module):
    
    #Initializing the class
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        #Temperature
        self.T = T
        
    #Brain (neural network) output signals forward to body
    def forward(self, outputs):
        #softmax outputs and multiply by temperature
        probs = F.softmax(outputs * self.T)
        
        #Actions to play
        actions =probs.multinomial(1)
        
        return actions
    

#Making the AI
class AI:
    
    #Initializing class
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    
    #Calling both forward methods from brain and body
    def __call__(self, inputs):
        #Reshaping input data
        input_data = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        
        #Putting data through the brain and taking the output
        output = self.brain(input_data)
        
        #Putting output into body and taking the actions
        actions = self.body(output)

        #Returning actions in right format
        return actions.data.numpy()



#Part 2 - Implementing Deep COnvolutional Q-Learning

#Getting the doom enviroment
#gym.make imports the enviroment
#image_preprocessing proccesses images coming in with 80 by 80 size in grayscale
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete('minimal')(gym.make('ppaquette/DoomCorridor-v0'))), 
                                               width = 80, height = 80, grayscale = True)
#Saves videos of AI playing doom into the videos folder
doom_env = gym.wrappers.Monitor(doom_env, 'videos', force = True)

#Getting number of actions from doom_enviroment
number_actions = doom_env.action_space.n


#Building an AI
#Creating an object of our CNN class
cnn = CNN(number_actions)
#Creating an object of our SoftmaxBoddy class and inputing temperature
softmax_body = SoftmaxBody(T = 1.0)
#Creating an object of our AI class and inputing the brain and body
ai = AI(cnn, softmax_body)


#Setting up Experiance Replay
#10 step learning with a capacity of 10,000
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
#Replay memory, create mini batches of 10 steps from 10,000 capacity
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

#Implementing Elgibility Trace
#n-step Q-learning (Not Asynchronous because we only have one agent)
#AKA sarsa?
#Training on batches
def eligibility_trace(batch):
    gamma = 0.99
    #Prediction
    inputs = []
    #Target
    targets = []
    
    #Going through the batch
    for series in batch:
        #Getting first and last transition of the series
        input_states = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        #Output signal
        output = cnn(input_states)
        
        #N-step Q-Learning
        #Cumulative reward
        #If last transition of the series is done, else get maximum of our Q values
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        
        #Steping from right to left up to the element before the last element (first element)
        for step in reversed(series[:-1]):
            #Update cumulative reward by multiplying by gamma and adding step reward
            cumul_reward = step.reward + gamma * cumul_reward
        
        #Get state of first transition
        state = series[0].state
        #Q value associated with the first step
        target = output[0].data
        #Target associated with action of first step is equal to the cumulative reward
        target[series[0].action] = cumul_reward
        
        #Append state and target to their lists
        inputs.append(state)
        targets.append(target)

    #returning inputs as a torch tensor and the targets stacked
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)


#Making the moving average on 100 steps
#Moving Average class
class MA:
    
    #Initializing class
    #Size we will use to compute moving average (100)
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    
    #Add cumulative reward to list of rewards
    def add(self, rewards):
        #If rewards is a list, add it to the current list of rewards
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        
        #If rewards is not a list, add reward by append
        else:
            self.list_of_rewards.append(rewards)
        
        #Makes sure list remains equal to given size
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    #Compute moving average of list of rewards
    def average(self):
        return np.mean(self.list_of_rewards)
    
#Creating an object of MA class
ma = MA(100)


#Training the AI
#Mean Square Error used for regression
loss = nn.MSELoss()
#Small learning rate to let AI explore more
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
#Number of epochs
nb_epochs = 100

#Iterating through number of epochs
for epoch in range(1, nb_epochs + 1):
    #Each epoch will be 200 runs of 10 steps
    memory.run_steps(200)
    
    #Generating batches of 128 from memory
    for batch in memory.sample_batch(128):
        #Putting batches through elgibility trace
        inputs, targets = eligibility_trace(batch)
        
        #Reshaping into pytorch variables
        inputs, targets = Variable(inputs), Variable(targets)
        
        #Get predictions from CNN (Brain)
        predictions = cnn(inputs)

        #Calculate loss error from predictions and targets
        loss_error = loss(predictions, targets)
        
        #Backpropagate back into CNN (Brain)
        #Initialize optimizer
        optimizer.zero_grad()
        
        #Applying backward propagation
        loss_error.backward()
        
        #Update weights with stochastic gradient decent
        optimizer.step()
    
    #Compute average reward
    #New Cumulative rewards of the steps
    rewards_steps = n_steps.rewards_steps()
    
    #Add new cumulative rewards to moving average object
    ma.add(rewards_steps)
    
    #Computing the average reward
    avg_reward = ma.average()
    
    #Printing avg reward for every epoch
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    #If we reach an avg_Reward greater or equal to 1000 we are sure we reached the vest
    if avg_reward >= 1000:
        print('Congratulations, your AI wins!')        
        #Stop training
        break

#Closing the doom enviroment
doom_env.close()
