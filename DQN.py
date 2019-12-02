"""
    File: DQN.py
    Author:	Jethjera, Silasant, Jonny Le

    Solution of Open AI gym environment "Cartpole-v0" using DQN and Pytorch.

    Modules:
        Network
        Learner
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# consider adding new layers, or change layer parameters
class Network(nn.Module):
    """ 
        This is the model network using primarily linear
        i/o connections
    """
    def __init__(self, num_states, num_actions, num_nodes):
        super(Network, self).__init__()
        # Initialize layers
        self.fc1 = nn.Linear(num_states, num_nodes)
        self.fc2 = nn.Linear(num_nodes, num_actions)
        # Normal distribution initialization to weights
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)   

    def forward(self, observation):
        """ 
            Feed forward our observation (forward propagation) and get the value of the actions
            given that we're in some set of states denoted by the observation
        """
        # Apply relu activation and feed forward
        observation = F.relu(self.fc1(observation))
        actions = self.fc2(observation)

        return actions


class Learner(object):
    """
        The constructor takes in arguments to instantiate the network
        and specify dimensions (number of states, actions, nodes), as well
        as other parameters for iterative learning.

        Fields:
            Q_eval   - the learner's estimate of the current set of states
            Q_target - the learner's estimate of the successor's set of states
            gamma    - discount factor so the learner has a choice of how to value
                        future rewards
            epsilon  - for greedy action selection / policy
            alpha    - the learning rate for the optimizer
            max_mem_size - to keep track of memory capacity
            action_space - corresponds to all possible actions for our learner
            memory_counter - for memory storage / indexing
            learn_step_counter - to keep track of how many times the learner has called 
                        the learn function for target network replacement
        
        Functions: 
            choose_action
            store_transition
            learn
    """
    def __init__(self, num_actions, num_states, num_nodes,
                alpha, gamma, epsilon, max_mem_size, action_space):
        self.Q_eval = Network(num_states, num_actions, num_nodes)
        self.Q_target = Network(num_states, num_actions, num_nodes)
        self.memory = np.zeros((max_mem_size, num_states * 2 + 2))
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_mem_size = max_mem_size
        self.action_space = action_space
        self.num_actions = num_actions
        self.num_states = num_states
        self.learn_step_counter = 0                                     
        self.memory_counter = 0                                         
        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=alpha)

    def choose_action(self, observation):
        """
            This function is used to take an action. It utilizes numpy to calculate a random 
            number to use for the epsilon-greedy action selection.
        """
        observation = T.unsqueeze(T.FloatTensor(observation), 0)
        # Epsilon-greedy policy
        if np.random.uniform() < self.epsilon:  
            # Get all of the Q values for the current state (forward prop)
            actions_value = self.Q_eval.forward(observation)

            # Take the optimal action 
            action = T.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.action_space == 0 else action.reshape(self.action_space)  # return the argmax index
        else:   
            # Choose a random action in the action space list
            action = np.random.randint(0, self.num_actions)
            action = action if self.action_space == 0 else action.reshape(self.action_space)

        return action

    def store_transition(self, state, action, reward, new_state):
        """
            This function stores memory transitions. It takes in the current state, the action taken,
            the reward received, and the resulting state.
        """
        # Compute index and store transition data
        index = self.memory_counter % self.max_mem_size
        self.memory[index, :] = np.hstack((state, [action, reward], new_state))

        # Increment counter for next index
        self.memory_counter += 1

    def learn(self, batch_size, target_cntr):
        """
            This function performs batch learning. It does random sampling of state transitions
            through the memory space to converge to an optimal strategy. It uses a target counter
            to keep track of how often we replace the target network
        """
        # Update target parameter and learn step counter
        if self.learn_step_counter % target_cntr == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.learn_step_counter += 1

        # Select a random sub-sample of memory and get batch
        index = np.random.choice(self.max_mem_size, batch_size)
        batch_mem = self.memory[index, :]
        batch_rewards = T.FloatTensor(batch_mem[:, self.num_states+1:self.num_states+2])
        batch_new_state = T.FloatTensor(batch_mem[:, -self.num_states:])
        batch_action = T.LongTensor(batch_mem[:, self.num_states:self.num_states+1].astype(int))
        batch_state = T.FloatTensor(batch_mem[:, :self.num_states])

        # Update current state and successor state after an action was taken
        q_eval = self.Q_eval(batch_state).gather(1, batch_action)
        q_next = self.Q_target(batch_new_state).detach()
        q_target = batch_rewards + self.gamma * q_next.max(1)[0].view(batch_size, 1)
        
        # Calculate loss function using MSE
        loss_func = nn.MSELoss()
        loss = loss_func(q_eval, q_target)

        # Back Propagation
        self.optimizer.zero_grad() # Zero out the gradient 
        loss.backward()
        self.optimizer.step()      # Step