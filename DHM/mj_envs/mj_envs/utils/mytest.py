import mj_envs
import gym


env = gym.make('hammer-v2')
env.reset()
for _ in range(10):
    env.render()
    action = env.action_space.sample()
    yang = env.step(action)  # take a random action

env.close()