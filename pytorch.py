import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
from skimage.color import rgb2gray
from skimage import transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as FN
import torchvision.transforms as TV

env = gym.make('SpaceInvaders-v0')


def preprocessing(frame):
    grayscaled_frame = rgb2gray(frame)
    cropped_frame = grayscaled_frame[8:-12,4:-12]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame,[110,84])
    return preprocessed_frame

def select_action(state):
    steps = 0 
    sample = random.random()
    epsilon_difference = final_epsilon + (start_epsilon - final_epsilon) * \
        math.exp(-1. * steps/decay_rate)
    steps = steps + 1
    if sample > epsilon_difference:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(actions)]])

#NEEDS TO BE FIX
#RuntimeError: size mismatch, m1: [1 x 22528], m2: [512 x 4] at c:\n\pytorch_1559129895673\work\aten\src\th\generic/THTensorMath.cpp:940
class DeepQLearning(nn.Module):

    def __init__(self, actions):
        super(DeepQLearning, self).__init__()
        self.convnet1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
       # self.batch1 = nn.BatchNorm2d(32)
        self.convnet2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
       # self.batch2 = nn.BatchNorm2d(64)
        self.convnet3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.batch3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, actions)

    def forward(self, x):
        x = x.float()/255
        x = FN.elu(self.convnet1(x))
        x = FN.elu(self.convnet2(x))
        x = FN.elu(self.convnet3(x))
        x = FN.elu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x.view(x.size(0), -1))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def save(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transitions(zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), 
                                            device=device, 
                                            dtype=torch.uint8)
    none_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])


    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(none_final_next_states).max(1)[0].detach()
    expected_Q_values = (next_state_values * gamma) + reward_batch

    #Huber Loss calculation
    loss = FN.smooth_l1_loss(state_action_values, expected_Q_values.unsqeeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

#NOT FINISHED TRAINING
#NEED TO FINISH def GET STATE
def train(env, num_episode, render=True):
    for episode in range(num_episode):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0
        for t in count():
            action = select_action(state)
            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward = reward + 1

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            memory.save(state, action, next_state, reward)

            state = next_state

            optimize_model()
            if done:
                break

        if episode % update_target == 0:
            target_net.load_state_dict(policy_net.state_dict())


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    actions = env.action_space.n
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    #Hyperparameters
    batch_size = 100
    gamma = 0.9
    start_epsilon = 0.9
    final_epsilon = 0.05
    decay_rate = 200
    update_target = 10
    num_episode = 25
    memory_size = 100000
    #
    policy_net = DeepQLearning(actions=4).to(device)
    target_net = DeepQLearning(actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    env = gym.make('SpaceInvaders-v0')

    memory = ReplayMemory(memory_size)

    train(env, num_episode)
    env.close()
    print('Complete')
