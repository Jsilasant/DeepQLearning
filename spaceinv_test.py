import gym

# Actions: Discrete(6)
# Box: Box(210, 160, 3)

# Random agent
env = gym.make('SpaceInvaders-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close()