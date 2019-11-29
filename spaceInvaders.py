import retro
import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import transform
from skimage.color import rgb2gray
import random
import matplotlib.pyplot as plt 
from collections import deque
import warnings

warnings.filterwarnings('ignore')

class PreAnalysis:
    def preprocessing(frame):
        gray = rgb2gray(frame)
        crop_frame = gray[8:-12,4:-12]
        normalize_frame = crop_frame/255.0
        preprocessing = transform.resize(normalize_frame,[110,84])
        return preprocessing

    def stacking_frames(stacked_frames, state, state_progress_new):
        frame = preprocessing(state)

        if state_progress_new:
            stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

            stacked_frames(frame)
            stacked_frames(frame)
            stacked_frames(frame)
            stacked_frames(frame)
            stacked_state = np.stack(stacked_frames, axis=2)

        else: 
            stacked_frames.append(frame)
            stacked_frames = np.stack(stacked_frames, axis=2)
        return stacked_frames, stacked_state

class DeepQNetwork:
    def __init__(self, state_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(ttf.float32, [None, self.action_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.convnet1 = tf.layers.conv2d(inputs = self.inputs_,
                                                      filters = 32,
                                                      )
def main():
    env = retro.make(game='SpaceInvaders-Atari2600')
    obs = env.reset()
    stack_size = 4
    state_size = [110,84,4]
    action_size = env.action_space.n
    learning_rate = 0.00025
    total_runs = 50
    max_movements = 50000
    batch_size = 64
    explore_begin = 1.0
    explore_stop = 0.1
    rate_of_decay = 0.00001
    gamma = 0.9
    pretrain_time = batch_size
    memory = 1000000
    training = False
    run_render = False
    
    while True:
        
        
        # action_space will by MultiBinary(16) now instead of MultiBinary(8)
        # the bottom half of the actions will be for player 1 and the top half for player 2
        obs, rew, done, info = env.step(env.action_space.sample())
        # rew will be a list of [player_1_rew, player_2_rew]
        # done and info will remain the same
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()