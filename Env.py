# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p, q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = [[p, q, r] for p in range(m) for q in range(t) for r in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for i in range(m+t+d)]
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 2:
            requests = np.random.poisson(12)
        elif location == 3:
            requests = np.random.poisson(4)
        elif location == 4:
            requests = np.random.poisson(7)
        elif location == 5:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""

        reward = (R* Time_matrix[state[0]][action[1]][state[1]][state[2]]) - ( C * (Time_matrix[state[0]][action[1]][state[1]][state[2]] + Time_matrix[state[0]][action[0]][state[1]][state[2]]))

        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""

        
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
