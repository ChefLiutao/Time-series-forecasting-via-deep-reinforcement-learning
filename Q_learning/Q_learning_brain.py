# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:53:22 2020

@author: ChefLiutao

Brain of Lookup Q-learning, e.g. the brain of Q-learning RL agent.
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







def choose_action(s,q_table,epsilon = 0.9):
    '''
    Args:
        s: state
        q_table: value table
    Return:
        action: chosen action
    '''
    state_action = q_table.loc[s,:]
    if (np.random.uniform() > epsilon) or ((state_action == 0).all()):
        action = np.random.choice(state_action.index)
    else:
        action = state_action.idxmax()
    return action

def build_q_table(n_states,actions):
    '''
    Args:
        n_states: dimension of state space
        action: all possible actions in action space
    Return:
        q_table: A DataFrame,index is state and column is action
    '''
    q_table = pd.DataFrame(data = np.zeros([n_states,len(actions)]),
                           columns = actions)
    return q_table

def get_env_feedback(s,a):
    if a == 'right':
        if s == 5.:
            r,s_ = 1.,'terminal'
        else:
            r,s_ = 0.,s+1
    else:
        if s == 0.:
            r,s_ = 0.,s
        else:
            r,s_ = 0.,s-1
    return r,s_


N_STATES = 6
ACTIONS = ['left','right']
MAX_EPISODES = 20
GAMMA = 0.9
ALPHA = 0.1
MAX_STEPS = 100

def rl():
    '''
    The main function of Q_learning
    '''
    q_table = build_q_table(N_STATES,ACTIONS)
    
    for episode in range(MAX_EPISODES):
        s = 0
        for step in range(MAX_STEPS):
            a = choose_action(s,q_table)
            r,s_ = get_env_feedback(s,a)
            q_current = q_table.loc[s,a]
            if s_ == 'terminal':
                q_target = r
            else:
                q_target = r + GAMMA*q_table.loc[s_,:].max()
            q_table.loc[s,a] += ALPHA*(q_target - q_current)
            
            if s_ == 'terminal':
                print('Episode %d - total steps: %d' %(episode,step+1))
                break
            s = s_
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print(q_table)
            
        
        
        
        
        
    
    
    
    
    