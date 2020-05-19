# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:04:48 2020

@author: ChefLiutao

The environment of 1-dimension teasure hunt game, of which both the state space
and action space are discrete.
"""

class One_D_teasure_env():
    def __init__(self):
        self.state_space = [0,1,2,3,4,5]
        self.action_space = ['left','right']
    
    def reset(self):
        '''
        初始化设置
        '''
        state = 0
        return state
    
    
    def step(self,state,action):
        '''
        定义环境动力学，输入s和a，返回next_state和reward
        '''
        if action == 'right':
            if state == 4:
                reward = 1
                next_state = 'terminal'
            else:
                reward = 0
                next_state = state + 1
        elif action == 'left':
            reward = 0
            if state == 0:
                next_state = state
            else:
                next_state = state - 1
        return next_state,reward
                

                
