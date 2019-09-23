import gym
import time

env = gym.make('Breakout-v0')
env.reset()

for _ in range(1000):
    env.render()
    obs, rew, done, info = env.step(env.action_space.sample())
    if rew != 0:
        print("Reward", rew)
    time.sleep(0.05)
    if done:
        break
