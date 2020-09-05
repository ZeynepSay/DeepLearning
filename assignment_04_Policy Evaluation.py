#!/usr/bin/env python
# coding: utf-8

# # Robot Learning assignmnet 04
# # Team members: Roberto Cai / Ramesh Kumar

# In[1]:
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# In[2]:


class GridWorld(object):
    def __init__(self, episodes):
        self.grid_world = np.array([[1,0,0,0,0,0,0],
                                    [0,0,0,1,1,1,0],
                                    [2,1,1,0,1,0,0],
                                    [0,0,1,0,1,0,3],
                                    [1,1,0,1,0,1,0]])# 0: empty, 1: W, 2: G, 3: S
        # contains all possible rewards
        self.rewards = {0:-1, 1:-20, 2:100, 3:-1}
        
        # Col 0 contains actions
        # col 1 contains probabilities
        # col 2 and 3 are used for obtaining next state
        # col 4 unicode values for arrows
        self.actions = np.array([[0,0, -1, 0, 8593],      # up
                                 [1,0, -1, 1, 8599],      # upper right
                                 [2,0, 0, 1, 8594],       # right
                                 [3,0, 1, 1, 8600],       # lower right
                                 [4,0, 1, 0, 8595],       # down
                                 [5,0.25, 1, -1, 8601],   # lower left
                                 [6,0.5, 0, -1, 8592],    # left
                                 [7,0.25, -1, -1, 8598]]) # upper left
        self.n_actions = 8
        self.transition_probability = np.array([0.15,0.7,0.15]) # left diagonal, desired direction, right diagonal
        self.episodes = episodes
        self.rows = 5
        self.cols = 7
        self.gamma = 1
        self.epsilon = 0.1
        self.alpha = 1
        self.states = self.rows*self.cols
    
    # get the expected value of a state
    def get_value(self,row,col):
        index = row*self.rows + col
        return self.V[index]
    
    # get the wind direction for non deterministic action
    # 0: deviation to right, 1: desired action, 2: deviation to left
    def stochastic_action(self):
        rand = np.random.random()
        for i in range(3):
            totalsum = np.sum(self.transition_probability[:i+1])
            if rand < totalsum:
                direction = i
                break
        return direction
    
    # select an action acording to probabilities
    def get_action(self):
        rand = np.random.random()
        for i in range(self.n_actions):
            totalsum = np.sum(self.actions[:i+1,1])
            if rand < totalsum:
                action = i
                break
        direction = self.stochastic_action()
        if direction == 0: # deviation to the right
            if action < 7:
                action += 1
            else:
                action = 0
        if direction == 2: # deviation to the left
            if action > 0:
                action += -1
            else:
                action = 7
        return action
    
    # choose action with max action value
    def get_greedy_action(self, row, col):
        index = row*7 + col
        a_index = np.argmax(self.Q[index,:])
        return a_index
    
    # obtain next state according to current state and selected action
    def get_next_state(self, row, col, action):
        if action == 0: # up
            if row == 0:
                r = row
                c = col 
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        elif action == 1: # upper right
            if row == 0 and col != 6:
                r = row
                c = col + self.actions[action,3]
            elif row == 0 and col == 6:
                r = row
                c = col
            elif row != 0 and col == 6:
                r = row + self.actions[action,2]
                c = col
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        elif action == 2: # left
            if col == 6:
                r = row
                c = col 
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        elif action == 3: # lower right
            if row == 4 and col != 6:
                r = row
                c = col + self.actions[action,3]
            elif row == 4 and col == 6:
                r = row
                c = col
            elif row != 4 and col == 6:
                r = row + self.actions[action,2]
                c = col
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        elif action == 4: # down
            if row == 4:
                r = row
                c = col 
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        elif action == 5: # lower left
            if row == 4 and col != 0:
                r = row
                c = col + self.actions[action,3]
            elif row == 4 and col == 0:
                r = row
                c = col
            elif row != 4 and col == 0:
                r = row + self.actions[action,2]
                c = col
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        elif action == 6: # left
            if col == 0:
                r = row
                c = col 
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        elif action == 7: # upper left
            if row == 0 and col != 0:
                r = row
                c = col + self.actions[action,3]
            elif row == 0 and col == 0:
                r = row
                c = col
            elif row != 0 and col == 0:
                r = row + self.actions[action,2]
                c = col
            else:
                r = row + self.actions[action,2]
                c = col + self.actions[action,3]
        return int(r), int(c)
    
    # obtain reward of next state
    def get_reward(self,row, col):
        return self.rewards[self.grid_world[row,col]]
    
    
    def td0_policy_evaluation(self):
        # initialize V values
        self.V = np.zeros(self.states)
        for i in range(self.episodes):
            # initial state
            row = 3
            col = 6
            
            is_terminal = True
            # loop until a terminal state is found
            while is_terminal:
                
                a = self.get_action()
                
                nrow, ncol = self.get_next_state(row, col, a)

                reward = self.get_reward(nrow, ncol)
                
                # calculate index of current state
                oind = (row)*7 + col
                # calculate index of new state
                nind = (nrow)*7 + ncol
                
                # update V 
                self.V[oind] = self.V[oind] + self.alpha*(reward + self.gamma*self.V[nind] - self.V[oind])
                
                row = nrow
                col = ncol
                
                # End if next state is W or G
                if self.grid_world[row,col] in (1,2):
                    is_terminal = False
        return self.V
        
    def q_learning(self):
        # initialize V, Q and policy values
        self.V = np.zeros(self.states)
        self.Q = np.zeros((self.states,self.n_actions))
        self.policy = []
        for i in range(self.episodes):
            # initial state
            row = 3
            col = 6
            
            is_terminal = True
            # loop until a terminal state is found
            while is_terminal:
                rand_a = np.random.random()
                # select e-greedy action 
                if rand_a < self.epsilon:
                    a = self.get_greedy_action(row,col)
                # select action acording to policy
                else:
                    a = self.get_action()
                
                nrow, ncol = self.get_next_state(row, col, a)
                
                reward = self.get_reward(nrow, ncol)
                
                # calculate index of current state
                oind = (row)*7 + col
                # calculate index of new state
                nind = (nrow)*7 + ncol
                
                # get the action with max action value
                q_ind = np.argmax(self.Q[nind,:])
                
                # update Q
                self.Q[oind,a] = self.Q[oind,a] + self.alpha*(reward + self.gamma*self.Q[nind,q_ind] - self.Q[oind,a])
                
                row = nrow
                col = ncol
                
                # End if next state is W or G
                if self.grid_world[row,col] in (1,2):
                    is_terminal = False
        # compute resulting V and policy
        for i in range(self.states):
            row = int(i / 7)
            col = i % 7
            q_ind = np.argmax(self.Q[i,:])
            if self.grid_world[row,col] in (1,2):
                self.policy.append('-1')
            else:
#                 print(int(self.actions[q_ind,4]))
                self.policy.append(chr(int(self.actions[q_ind,4])))
            self.V[i] = self.Q[i,q_ind]
            
        return self.Q, self.V, self.policy
                


# # Exercise 4.1

# In[3]:


grid = GridWorld(500)
v = grid.td0_policy_evaluation()
v.shape = (5,7)

v=pd.DataFrame(v)




# # Exercise 4.2

# In[3]:


grid = GridWorld(5000)
q, v, policy = grid.q_learning()
v.shape = (5,7)
pd.DataFrame(v)


# In[4]:


policy = np.asarray(policy)
policy.shape = (5,7)
pd.DataFrame(policy)




# In[ ]:




# In[ ]:




