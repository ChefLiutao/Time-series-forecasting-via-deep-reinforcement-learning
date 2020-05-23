# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:54:17 2020

@author: ChefLiutao

To test DQN using Cartpole-v0 environment in package [gym].
"""
import gym
from DQN_agent import Nature_DQN

env = gym.make('CartPole-v0')  
env = env.unwrapped   #Lift some restrictions

N_FEATURES = 4
N_ACTIONS = 2
MAX_EPISODES = 1000
MAX_STEPS = 500
dqn = Nature_DQN(4,2,16,0.003)

for episode in range(MAX_EPISODES):
    s = env.reset()
    for step in range(MAX_STEPS):
        a = dqn.epsilon_choose_a(s)
        s_,r,done,info = env.step(a)
        done = 0 if done else 1
        dqn.store_transition(s,a,r,s_,done)
        dqn.learn()
        if (done == 0) or (step == 499):
            print('Episode %d:%d'%(episode,step+1))
            break
        s = s_

dqn.plot_loss()

