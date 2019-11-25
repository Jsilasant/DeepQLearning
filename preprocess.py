import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray

# Box: Box(210, 160, 3)

# Grayscale and crop frames
# (210, 160, 3) -> (188, 144)
def preprocess_frame(frame):
   grayscaled_frame = rgb2gray(frame)
   cropped_frame = grayscaled_frame[8:196,2:146]
   return cropped_frame

# View Original vs Preprocessed
def main():
   env = gym.make('SpaceInvaders-v0')
   observation = env.reset()
   first_frame = env.render(mode='rgb_array')
   preprocessed = preprocess_frame(first_frame)
   fig, axes = plt.subplots(1, 2)
   ax = axes.ravel()

   ax[0].imshow(first_frame)
   ax[0].set_title("Original")
   ax[1].imshow(preprocessed, cmap="gray")
   ax[1].set_title("Grayscale + Cropped")

   fig.tight_layout()
   plt.show()

if __name__ == '__main__':
   main()
