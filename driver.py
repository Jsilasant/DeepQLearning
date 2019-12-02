"""
    File: driver.py
    Author:	Jethjera, Silasant, Jonny Le

    This is the driver program to instantiate and test the DQN algorithm
    against the OpenAI gym environment "CartPole-v0." It instantiates the
    environment, the DQN learner, and does iterative training to print results.
    
    Modules:
        None
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DQN import Learner, Network

SHOW_TRAINING = False # Flag to show visuals / rendering process

if __name__ == '__main__':

    # Instantiate environment
    env = gym.make('CartPole-v0').unwrapped
    action_space = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

    NUM_EPS = 300               # max number of episodes for training iteration
    MOD_X_R = 0.8                # modify reward by x
    MOD_TH_R = 0.5               # modify reward by theta
  
    # Instantiate learner
    dqn_agent = Learner(num_actions=env.action_space.n, num_states=env.observation_space.shape[0], 
                            num_nodes=50, alpha=1e-3, gamma=0.9, epsilon=0.95, 
                            max_mem_size=2000, action_space=action_space)

   # Training Iteration
    score_list = []
    episodes = range(NUM_EPS)
    for episode in episodes:

        observation = env.reset()
        ep_score = 0 # reset score counter

        while True:
            if SHOW_TRAINING:
                env.render()
            action = dqn_agent.choose_action(observation)

            # Take action, recieve new observation and reward
            new_obs, reward, done, info = env.step(action)

            # Update reward based off new observation
            x, _, theta, _ = new_obs
            reward_value1 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - MOD_TH_R
            reward_value2 = (env.x_threshold - abs(x)) / env.x_threshold - MOD_X_R
            reward = reward_value1 + reward_value2

            # Store transition data
            dqn_agent.store_transition(observation, action, reward, new_obs)

            # Update score for current episode
            ep_score += reward
             
            # Invoke learn function
            dqn_agent.learn(batch_size=32, target_cntr=100)

            if done:
                # Append current score to list
                score_list.append(ep_score)
                print('Episode: ', episode+1,'| Score: ', round(ep_score, 2))
                break
            observation = new_obs

    # Calculate average score and print
    avg_score = sum(score_list)/NUM_EPS
    print('Average score: ', round(avg_score, 2))

    # Plot visualizations
    plt.plot(score_list)
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.savefig(str(NUM_EPS)+'-ep.pdf')

    # Close env
    env.close()