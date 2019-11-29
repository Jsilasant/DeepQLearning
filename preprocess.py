import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import tensorflow as tf
import retro                 
from skimage import transform # Help us to preprocess the frames
import matplotlib.pyplot as plt # Display graphs
from collections import deque# Ordered collection with ends
import random
import warnings
# Box: Box(210, 160, 3)

# Grayscale and crop frames
# (210, 160, 3) -> (188, 144)
def preprocess_frame(frame):
   grayscaled_frame = rgb2gray(frame)
   cropped_frame = grayscaled_frame[8:-12,4:-12]
   normalized_frame = cropped_frame/255.0
   preprocess_frame = transform.resize(normalized_frame,[110,84])
   return cropped_frame

# View Original vs Preprocessed
def main():
   env = retro.make(game='SpaceInvaders-Atari2600')

   print("The size of our frame is: ", env.observation_space)
   print("The action size is : ", env.action_space.n)

   possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames
    
if __name__ == '__main__':
   main()
