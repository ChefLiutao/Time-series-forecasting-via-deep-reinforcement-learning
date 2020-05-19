# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:38:00 2020

@author: ChefLiutao

To test the effectiveness of Q-learning in 1D teasure hunt game.
"""

from Q_learning_brain import Tabular_q_learning
from Teasure_hunt_env import One_D_teasure_env

######################################################

env = One_D_teasure_env()
states = env.state_space
actions = env.action_space

LEARNING_RATE = 0.1
EPSILON = 0.9
GAMMA = 0.9
MAX_EPISODES = 20
MAX_STEPS = 100

q_learning = Tabular_q_learning(states,actions,LEARNING_RATE,EPSILON,GAMMA)

######################################################

def rl_learn_loop():
    for episode in range(MAX_EPISODES):
        s = env.reset()
        for step in range(MAX_STEPS):
            a = q_learning.epsilon_choose_action(s)
            s_,r = env.step(s,a)
            q_learning.learn(s,a,r,s_)
            
            if s_ == 'terminal':
                print('Episode %d - total steps - %d'%(episode,step+1))
                break
            
            s = s_
            

if __name__ == '__main__':
    rl_learn_loop()
    print(q_learning.q_table)