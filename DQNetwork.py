import tensorflow as tf 
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import gym
import tflearn
import numpy as np
import random
import threading
import time

game ='SpaceInvaders-v0'
testing = False
show_training = True
checkpoint_path = 'qlearning.tflearn.ckpt'
checkpoint_interval = 2000
num_eval_episodes = 100




def DeepQlearning(actions, repeat_action):
    inputs = tf.placeholder(tf.float32, [None, repeat_action, 84,84])
    net = tf.transpose(inputs, [0,2,3,1])
    net = tflearn.conv_2d(net, 64, 8, strides=4, activation='elu')
    net = tflearn.conv_2d(net, 32, 8, strides=4, activation='elu')
    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
    net = tflearn.fully_connected(net, 256, activation='elu')
    q_values = tflearn.fully_connected(net, actions)
    return inputs, q_values

class Environment(object):
    def __init__(self, gym_env, repeat_action):
        self.env = gym_env
        self.repeat_action = repeat_action
        self.gym_actions = range(gym_env.action_space.n)
        self.state_buffer = deque()

    def get_start_state(self):
        self.state_buffer = deque
        new_data = self.env.reset
        new_data = self.get_preprocessed_frame(new_data)
        new_stack = np.stack([new_data for i in range(self.action_repeat)], axis=0)

        for i in range(self.repeat_action - 1):
            self.state_buffer.append(new_data)
        return new_stack

    def get_preprocessed_frame(self, observation):
        return resize(rgb2gray(observation),(110,84))[8:-12,4:-12]

def eplisons():
    eplison = np.array([.1,.01,.5])
    probability = np.array([0.4, 0.3, 0.3])
    return np.random.choice(eplisons, 1, p=list(probability))[0]

def actor_progression(thread_id, env, session, actions, summary_ops, saver):
    max_steps = 5000
    current_steps = 0
    final_eplison =  eplisons()
    starting_eplison = 1.0
    epsilon = 1.0
    epsilon_timesteps = 40000
    gamma = 0.99
    
    env = Environment(gym_env=env, repeat_action= repeat_action)
    state_batch = []
    action_batch = []
    gamma_batch = []

  
    print("Thread " + str(thread_id) + " - Final epsilon: " + str(final_epsilon))
    
    time.sleep(3*thread_id)
    t = 0
    while current_steps < max_steps:
    
        state_steps = env.get_start_state()
        terminal = False
        episode_rewards = 0
        episode_max_q = 0
        episode_steps = 0

        while True:

            readout_steps = q_values.eval(session=session, feed_dict={s:[state_steps]})


        action_steps = np.zeros([actions])
        if random.random() <= epsilon:
            action_index = random.randrange(actions)
        else:
            action_index = np.argmax(readout_steps)
        action_steps[action_index] = 1

        if epsilon > final_eplison:
            epsilon = ((starting_eplison - final_eplison)/ epsilon_timesteps) - epsilon

        state_steps1, reward_steps, terminal, info = env.step(action_index)
        readout_j =  target_q_values.eval(session= session, feed_dict={s:[state_steps]})

        clipped_reward_step = np.clip(reward_steps, -1,1)
        if terminal:
            gamma_batch.append(clipped_reward_step)
        else:
            gamma_batch.append(clipped_reward_step + gamma * np.max(readout_j))

        action_batch.append(action_steps)
        state_batch.append(state_steps)
        state_steps = state_steps1
        current_steps = current_steps + 1
        t = t+1

        episode_steps = episode_steps + 1
        episode_rewards = reward_steps + 1
        episode_max_q = np.max(readout_steps) + 1

        if t % checkpoint_interval == 0:
            saver.save(session, "learning.ckpt", global_step=t)

        if terminal:
            stats = [episode_rewards, episode_max_q/float(episode_steps), epsilon]
            print("| Step", t,
                        "| Reward: %.2i" % int(episode_reward), " Qmax: %.4f" %
                        (episode_max_q/float(episode_steps)),
                        " Epsilon: %.5f" % epsilon, " Epsilon progress: %.6f" %
                        (t/float(epsilon_timesteps)))
            break
    
def train(session, actions, saver):
    env = gym.make('SpaceInvaders-v0')
    while True:
        if show_training:
                env.render()



def get_num_actions():
    env = gym.make('SpaceInvaders-v0')
    actions = env.action_space.n
    return actions
    
def main(_):
    with tf.Session() as session:
        actions = get_num_actions()
        #graph_ops = build_graph(num_actions)
        saver = tf.train.Saver(max_to_keep=5)

        if testing:
            evaluation(session, saver)
        else:
            train(session, actions, saver)

if __name__ == "__main__":
    tf.app.run()