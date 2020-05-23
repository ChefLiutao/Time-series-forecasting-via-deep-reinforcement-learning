# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:05:29 2020

@author: ChefLiutao
"""
import tensorflow as tf
import numpy as np
from collections import deque
import random

class Nature_DQN():
    def __init__(
            self,
            n_features,
            n_actions,
            n_hidden,
            learning_rate,
            epsilon = 0.9,
            gamma = 0.9,
            memory_size = 1000,
            batch_size = 128,
            epsilon_increment = 0.0001):
        
        self.n_features = n_features       #dimension of state
#        self.actions = actions             #all possible actions
        self.n_actions = n_actions      #dimension of action space
        self.n_hidden = n_hidden           #hidden neurons of Q network
        self.lr = learning_rate            #for Current Q network update
        self.epsilon = epsilon             #e-greed
        self.gamma = gamma                 #reward discount rate
        self.memory_size = memory_size
        self.memory = deque(maxlen = memory_size)
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.loss_history = []
        self.learn_step_counter = 0
        
        self.s = tf.placeholder(dtype = tf.float32,shape = [None,self.n_features])
        self.a = tf.placeholder(tf.int32,shape = [None,])
        self.s_ = tf.placeholder(tf.float32,shape = [None,self.n_features])
        self.r = tf.placeholder(tf.float32,shape = [None,])
        self.done = tf.placeholder(tf.float32,shape = [None,])  #denote if the s_ is terminal:0→terminal,1→non-terminal
        
        self.q_current = self.build_current_net()  # self.s → self.q_current; self.s_ → self.q_next
        self.q_next = self.build_target_net()
        
        self.current_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope = 'current_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope = 'target_net')
        
        self.params_replace = [tf.assign(t,e) for t,e in zip(self.target_params,self.current_params)]
        
        self.q_sa = tf.gather_nd(self.q_current,indices = tf.stack(
                [tf.range(tf.shape(self.a)[0],dtype = tf.int32),self.a],
                axis = 1))             #[tf.shape(self.a)[0],]
        
        self.q_s_a_ = tf.reduce_max(self.q_next,axis = 1)
        
        with tf.variable_scope('loss'):
            self.q_target = (self.r + self.gamma*self.q_s_a_)*self.done
            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q_sa))
            self.train_op = self.optimizer.minimize(self.loss)
            
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def build_current_net(self):
        '''
        Build Two Q neworks:current Q network and target Q network.
        Target Q network is respnsible for calculating Q(s_).
        Current Q network is responsible to estimate Q(s)
        '''
        # Current Q Network    ------------------------------------------
#        self.s = tf.placeholder(shape = [None,self.n_features])
#        self.q_target = tf.placeholder(shape = [None,len(self.actions)])
        
        with tf.variable_scope('current_net'):
            w_init = tf.random_normal_initializer(0,0.1)
            b_init = tf.constant_initializer(0.1)
            w1 = tf.get_variable(name = 'w1',
                                     shape = [self.n_features,self.n_hidden],
                                     initializer = w_init,
                                     trainable = True)
            b1 = tf.get_variable(name = 'b1',
                                     shape = [self.n_hidden],
                                     initializer = b_init,
                                     trainable = True)
    
#            tf.add_to_collection('current_params',w1)
#            tf.add_to_collection('current_params',b1)
                
            hidden = tf.nn.relu(tf.matmul(self.s,w1) + b1)
                
            w2 = tf.get_variable(name = 'w2',
                                     shape = [self.n_hidden,self.n_actions],
                                     initializer = w_init,
                                     trainable = True)
            b2 = tf.get_variable(name = 'b2',
                                     shape = [self.n_actions],
                                     initializer = b_init,
                                     trainable = True)
#            tf.add_to_collection('current_params',w2)
#            tf.add_to_collection('current_params',b2)
                
            self.q_current = tf.matmul(hidden,w2) + b2
        return self.q_current
    
    def build_target_net(self):
        w_init = tf.random_normal_initializer(0,0.1)
        b_init = tf.constant_initializer(0.1)
        # Target Q Network  ------------------------------------------------
#        self.s_ = tf.placeholder(shape = [None,self.n_features])
        with tf.variable_scope('target_net'):
            w1 = tf.get_variable(name = 'w1',
                                     shape = [self.n_features,self.n_hidden],
                                     initializer = w_init,
                                     trainable = False)
            b1 = tf.get_variable(name = 'b1',
                                     shape = [self.n_hidden],
                                     initializer = b_init,
                                     trainable = False)
#           tf.add_to_collection('target_params',w1)
#           tf.add_to_collection('target_params',b1)
            hidden = tf.nn.relu(tf.matmul(self.s_,w1) + b1)
            
            w2 = tf.get_variable(name = 'w2',
                                     shape = [self.n_hidden,self.n_actions],
                                     initializer = w_init,
                                     trainable = False)
            b2 = tf.get_variable(name = 'b2',
                                     shape = [self.n_actions],
                                     initializer = b_init,
                                     trainable = False)
#            tf.add_to_collection('target_params',w2)
#            tf.add_to_collection('target_params',b2)
        
            self.q_next = tf.matmul(hidden,w2) + b2
        return self.q_next
               
                
    def epsilon_choose_a(self,state):
        state = np.reshape(state,[-1,self.n_features])
        if np.random.uniform() < self.epsilon:
            state_action = self.sess.run(self.q_current,
                                         feed_dict = {self.s:state})
            action = np.argmax(state_action)
        else:
            action = np.random.choice(np.arange(self.n_actions))
        return action
    
    
    def store_transition(self,state,action,reward,next_state,is_done):
        state,next_state = state[np.newaxis,:],next_state[np.newaxis,:]
        action,reward,is_done = np.array(action),np.array(reward),np.array(is_done)
        action = action.reshape([1,1])
        reward = reward.reshape([1,1])
        is_done = is_done.reshape([1,1])
        
        transition = np.concatenate((state,action,reward,next_state,is_done),axis = 1)
        self.memory.append(transition[0,:])
    
    
    def learn(self):
        if len(self.memory) == self.memory_size:
            if self.learn_step_counter % 500 == 0:
                self.sess.run(self.params_replace)
            self.learn_step_counter += 1
            
            batch = np.array(random.sample(self.memory,self.batch_size))
            batch_s = batch[:,:self.n_features]
            batch_a = batch[:,self.n_features:(self.n_features + 1)][:,0]
            batch_r = batch[:,(self.n_features + 1):(self.n_features + 2)][:,0]
            batch_s_ = batch[:,(self.n_features + 2):(self.n_features*2 + 2)]
            batch_done = batch[:,-1]
            train_op,loss = self.sess.run((self.train_op,self.loss),
                                          feed_dict = {self.s:batch_s,self.a:batch_a,
                                                       self.s_:batch_s_,self.r:batch_r,
                                                       self.done:batch_done})
            self.loss_history.append(loss)
            
            self.epsilon = self.epsilon + self.epsilon_increment if (
                    self.epsilon + self.epsilon_increment) < 1. else 1.
            
    
    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.loss_history)),self.loss_history,'-')
