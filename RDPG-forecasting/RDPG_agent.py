# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:52:21 2020

@author: ChefLiutao

The agent of RL algorithm Recurrent Detrministic Policy Gradient.

The Actor NNs are deployed as three-layer Fully-Connected NN.

The Critic NNs are deployed as RNN.
"""
import tensorflow as tf
import numpy as np
from collections import deque
import random

class RDPG():
    def __init__(self,
                 n_features,
#                 n_actions,
                 a_low,
                 a_high,
                 learning_rate_actor,
                 learning_rate_critic,
                 n_actor_hidden,
                 n_critic_hidden,
                 gamma = 0.9,
                 noise_varience = 3,
                 soft_replace = 0.1,
                 memory_size = 1000,
                 batch_size = 128):
        self.n_features = n_features             #dimension of states
#        self.n_actions = n_actions        
        self.a_low = a_low                       #The low bound of action sapce
        self.a_high = a_high                     #The high bound of action space
        self.lr_a = learning_rate_actor          #Learning rate of Actor NN
        self.lr_c = learning_rate_critic         #Learning rate of Critic NN
        self.n_actor_hidden = n_actor_hidden     #Number of hidden layer neurons in Actor
        self.n_critic_cells = n_critic_hidden   #Number of hidden layer neurons in Critic
        self.gamma = gamma                       #Reward discount rate
        self.noise_var = noise_varience          #Variance of output action distribution
        self.soft_replace = soft_replace         #Update speed of target networks
        self.memory_size = memory_size           #Size of experience replay buffer
        self.memory = deque(maxlen = self.memory_size)   #Experience replay buffer
        self.batch_size = batch_size                     
        
        self.s = tf.placeholder(dtype = tf.float32,shape = [None,self.n_features])
        self.s_ = tf.placeholder(dtype = tf.float32,shape = [None,self.n_features])
        self.r = tf.placeholder(dtype = tf.float32,shape = [None,])
        self.done = tf.placeholder(dtype = tf.float32,shape = [None,]) # 0 if s_ == terminal else 1
        
        self.a = self.build_Actor1()
        self.a_ = self.build_Actor2()
        self.q_sa = self.build_Critic1()      #shape:[None,] 
        self.q_s_a_ = self.build_Critic2()    #shape:[None,]
        
        self.curr_a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope = 'Actor/Current')
        self.targ_a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope = 'Actor/Target')
        self.curr_c_params= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope = 'Critic/Current')
        self.targ_c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope = 'Critic/Target')
        
        # Soft replace of Targets NN parameters
        self.replace_a_params = [tf.assign(t,(1-self.soft_replace)*t + self.soft_replace*e) \
                                 for (t,e) in zip(self.targ_a_params,self.curr_a_params)]
        self.replace_c_params = [tf.assign(t,(1-self.soft_replace)*t + self.soft_replace*e) \
                                 for (t,e) in zip(self.targ_c_params,self.curr_c_params)]
        
        self.td_error = self.r + self.gamma*self.q_s_a_ - self.q_sa
        self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
        self.actor_loss = -tf.reduce_mean(self.q_sa)
        
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_a).minimize(self.actor_loss,
                                                    var_list = self.curr_a_params)
        self.critic_train_op = tf.train.AdamOptimizer(self.lr_c).minimize(self.critic_loss,
                                                     var_list = self.curr_c_params)
        
        self.learn_step_counter = 0
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    
    def build_Actor1(self):
        '''
        Building Current Actor network.
        '''
        with tf.variable_scope('Actor/Current'):
            w_init = tf.random_normal_initializer(0,0.1)
            b_init = tf.constant_initializer(0.1)
            w1 = tf.get_variable(name = 'w1',shape = [self.n_features,self.n_actor_hidden],
                                 dtype = tf.float32,initializer = w_init,
                                 trainable = True)
            b1 = tf.get_variable('b1',shape = [self.n_actor_hidden,],
                                 dtype = tf.float32,initializer = b_init,
                                 trainable = True)
            w2 = tf.get_variable('w2',shape = [self.n_actor_hidden,1],
                                 dtype = tf.float32,initializer = w_init,
                                 trainable = True)
            b2 = tf.get_variable('b2',shape = [1,],
                                 dtype = tf.float32,initializer = b_init,
                                 trainable = True)
            hidden = tf.nn.relu(tf.matmul(self.s,w1) + b1)
            a = tf.matmul(hidden,w2) + b2
        return a[:,0]
#            return np.clip(np.random.normal(a,self.noise_var),self.a_low,self.a_high)
    
    def build_Actor2(self):
        '''
        Building Target Actor network.
        '''
        with tf.variable_scope('Actor/Target'):
            w_init = tf.random_normal_initializer(0,0.1)
            b_init = tf.constant_initializer(0.1)
            w1 = tf.get_variable('w1',shape = [self.n_features,self.n_actor_hidden],
                                 dtype = tf.float32,initializer = w_init,
                                 trainable = False)
            b1 = tf.get_variable('b1',shape = [self.n_actor_hidden,],
                                 dtype = tf.float32,initializer = b_init,
                                 trainable = False)
            w2 = tf.get_variable('w2',shape = [self.n_actor_hidden,1],
                                 dtype = tf.float32,initializer = w_init,
                                 trainable = False)
            b2 = tf.get_variable('b2',shape = [1,],
                                 dtype = tf.float32,initializer = b_init,
                                 trainable = False)
            hidden = tf.nn.relu(tf.matmul(self.s_,w1) + b1)
            a_ = tf.matmul(hidden,w2) + b2
        return a_[:,0]
    
    def build_Critic1(self):
        '''
        Building Current Critic network.
        '''
        with tf.variable_scope('Critic/Current'):
            w_init = tf.random_normal_initializer(0,0.1)
            b_init = tf.constant_initializer(0.1)
            
            rnn_cell = tf.contrib.rnn.BasicRNNCell(self.n_critic_cells)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float64)
            s = tf.cast(tf.expand_dims(self.s,axis = 1),tf.float64)
            
            outputs, self.final_state = tf.nn.dynamic_rnn(
                    cell = rnn_cell, inputs = s, 
                    initial_state = self.init_state, time_major = True)
            cell_out = tf.cast(tf.reshape(outputs, [-1, self.n_critic_cells]),tf.float32)
            
            a_out = tf.layers.dense(self.a[:,np.newaxis],self.n_critic_cells,trainable = True)
            q_sa = tf.layers.dense(cell_out + a_out,1,tf.nn.relu,
                                   kernel_initializer = w_init,
                                   bias_initializer = b_init,trainable = True)

        return q_sa[:,0]
            
    
    def build_Critic2(self):
        '''
        Building Target Critic network.
        '''
        with tf.variable_scope('Critic/Target'):
            w_init = tf.random_normal_initializer(0,0.1)
            b_init = tf.constant_initializer(0.1)
            
            rnn_cell = tf.contrib.rnn.BasicRNNCell(self.n_critic_cells)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float64)
            s_ = tf.cast(tf.expand_dims(self.s_,axis = 1),tf.float64)
            
            outputs, self.final_state = tf.nn.dynamic_rnn(
                    cell = rnn_cell, inputs = s_, 
                    initial_state = self.init_state, time_major = True)
            cell_out = tf.cast(tf.reshape(outputs, [-1, self.n_critic_cells]),tf.float32)
            
            a_out = tf.layers.dense(self.a_[:,np.newaxis],self.n_critic_cells,trainable = False)
            q_s_a_ = tf.layers.dense(cell_out + a_out,1,tf.nn.relu,
                                   kernel_initializer = w_init,
                                   bias_initializer = b_init,trainable = False)

        return q_s_a_[:,0]         
    
    def choose_action(self,state):
        state = np.reshape(state,[-1,self.n_features])
        action = self.sess.run(self.a,feed_dict = {self.s:state})
        return action
    
    def store_transition(self,state,action,reward,next_state):
        state,next_state = state[np.newaxis,:],next_state[np.newaxis,:]
        action,reward = np.array(action),np.array(reward)
        action = np.reshape(action,[1,-1])
        reward = np.reshape(reward,[1,-1])
#        is_done = np.reshape(is_done,[1,-1])
        
        transition = np.concatenate((state,action,reward,next_state),axis = 1)
        self.memory.append(transition[0,:])
    
    def learn(self):
        if len(self.memory) == self.memory_size:
            if self.learn_step_counter % 200 == 0:
                self.sess.run((self.replace_a_params,self.replace_c_params))
            
            self.noise_var *= 0.999
                
            batch = np.array(random.sample(self.memory,self.batch_size))
            batch_s = batch[:,:self.n_features]
            batch_a = batch[:,self.n_features:(self.n_features + 1)][:,0]
            batch_r = batch[:,(self.n_features + 1):(self.n_features + 2)][:,0]
            batch_s_ = batch[:,(self.n_features + 2):(self.n_features*2 + 2)]
            
            self.sess.run(self.actor_train_op,feed_dict = {self.s:batch_s})
            self.sess.run(self.critic_train_op,feed_dict = {self.s:batch_s,
                                                            self.a:batch_a,
                                                            self.s_:batch_s_,
                                                            self.r:batch_r})

if __name__ == '__main__':
    rdpg = RDPG(5,0,1,0.03,0.01,30,30)