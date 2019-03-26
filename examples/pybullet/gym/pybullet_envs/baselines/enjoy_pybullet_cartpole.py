# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
from pybullet_envs.bullet.cartpole_bullet import CartPoleBulletEnv
from baselines import deepq
import time
import gym
import os
import inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


def main():
  env = gym.make('CartPoleBulletEnv-v1')
  act = deepq.load("cartpole_model.pkl")

  while True:
    obs, done = env.reset(), False
    print("obs")
    print(obs)
    print("type(obs)")
    print(type(obs))
    episode_rew = 0
    while not done:
      env.render()

      o = obs[None]
      aa = act(o)
      a = aa[0]
      obs, rew, done, _ = env.step(a)
      episode_rew += rew
      time.sleep(1. / 240.)
    print("Episode reward", episode_rew)


if __name__ == '__main__':
  main()
