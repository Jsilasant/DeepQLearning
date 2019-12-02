# DeepQLearning
How to run:

python driver.py

How to change parameters:
Parameters exist in Learner in driver. You can change the amount of alpha(learning_rate), gamma(discount rate), and epsilon(greedy-policy), 
and also change the maximum number of episodes
    "NUM_EPS = 300 "
    "dqn_agent = Learner(num_actions=env.action_space.n, num_states=env.observation_space.shape[0], 
                            num_nodes=50, alpha=1e-3, gamma=0.9, epsilon=0.95, 
                            max_mem_size=2000, action_space=action_space)"
                            
packages dependencies: "gym, pytorch, numpy, and matplotlib"
