# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:53:22 2020

@author: ChefLiutao

Brain of Lookup Q-learning, i.e. the brain of Q-learning RL agent.
"""
import numpy as np
import pandas as pd 

class Tabular_q_learning():
    def __init__(self,states,actions,learning_rate,epsilon = 0.9,gamma = 0.9):
        '''
        Args:
            states: A python list of all possible states in state space.
            actions: A python list of all possible actions in action space.
            learning_rate: Update speed of Q_table
            epsilon: A probablity that controls the trade-off between exploration and exploitation
            gamma: Reward discount rate
        '''
        self.state_space = states
        self.action_space = actions
        self.n_states = len(self.state_space)
        self.n_actions = len(self.action_space)
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha =learning_rate
        self.q_table = self.build_q_table()
      
        
    def build_q_table(self):
        q_table = pd.DataFrame(np.zeros([self.n_states,self.n_actions]),
                               index = self.state_space,
                               columns = self.action_space)
        return q_table
    
    
    def epsilon_choose_action(self,state):
        state_action = self.q_table.loc[state]
        if (np.random.uniform() > self.epsilon) or ((state_action == 0).all()):
            action = np.random.choice(state_action.index)
        else:
            action = state_action.idxmax()
        return action
    
    
    def learn(self,state,action,reward,next_state):
        q_current = self.q_table.loc[state,action]
        q_target = reward if (next_state == 'terminal') else (
                reward + self.gamma*self.q_table.loc[next_state].max())
        
        self.q_table.loc[state,action] += self.alpha*(q_target - q_current)
