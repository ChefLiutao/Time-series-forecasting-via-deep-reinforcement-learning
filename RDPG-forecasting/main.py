# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:23:52 2020

@author: ChefLiutao
"""
import pandas as pd
import numpy as np
from DataPreprocessing import normalization
from DataPreprocessing import build_s_a
from RDPG_agent import RDPG
import matplotlib.pyplot as plt

#####################  hyper parameters  ####################
N_FEATURES = 6
A_LOW = 0
A_HIGH = 1
LR_A = 0.001
LR_C = 0.003
N_ACTOR_HIDDEN = 30
N_CRITIC_HIDDEN = 30
MAX_EPISODES = 100
MAX_STEPS = 1000

GAMMA = 0.9                # 折扣因子
TAU = 0.1                 # 软更新因子
MEMORY_CAPACITY = 100000    #记忆库大小
BATCH_SIZE = 128            #批梯度下降的m
#############################################################

#Load data 
data_dir = '\\V6.csv'  #directory of time series data
data = pd.read_csv(data_dir,encoding = 'gbk')
data = data.iloc[:,0]

#Build state matrix and best action
state,action = build_s_a(data,N_FEATURES,1)

#Data split
SPLIT_RATE = 0.75
split_index = round(len(state)*SPLIT_RATE)
train_s,train_a = state[:split_index],action[:split_index]
test_s,test_a = state[split_index:],action[split_index:]

#Normalization
train_s_scaled,test_s_scaled = normalization(train_s,test_s)
A,B = train_a.max(),train_a.min()
train_a_scaled,test_a_scaled = (train_a-B)/(A-B),(test_a-B)/(A-B)

# Training
rdpg = RDPG(N_FEATURES,A_LOW,A_HIGH,LR_A,LR_C,N_ACTOR_HIDDEN,N_CRITIC_HIDDEN)
for episode  in range(MAX_EPISODES):
    index = np.random.choice(range(len(train_s_scaled)))
    s = train_s_scaled[index]
    ep_reward = 0
    
    for step in range(MAX_STEPS):
        a = rdpg.choose_action(s)
        r = -abs(a-train_a_scaled[index])
        ep_reward += r
        index += 1
        s_ = train_s_scaled[index]
        
        rdpg.store_transition(s,a,r,s_)
        rdpg.learn()
        
        if (index == len(train_s_scaled)-1) or (step == MAX_STEPS-1):
            print('Episode %d : %.2f'%(episode,ep_reward))
            break
        
        s = s_

# Testing
pred = []
for i in range(len(test_s_scaled)):
    state = test_s_scaled[i]
    action = rdpg.choose_action(state)
    pred.append(action)

pred = [pred[i][0] for i in range(len(test_s_scaled))]
pred = pd.Series(pred)
pred = pred*(A-B)+B
actual = pd.Series(test_a)

plt.scatter(pred,test_a,marker = '.')
plt.xlabel('Predicted Value')
plt.ylabel('Actual value')
plt.show()