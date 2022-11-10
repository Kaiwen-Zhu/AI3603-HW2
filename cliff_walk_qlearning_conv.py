# -*- coding:utf-8 -*-
# Train Q-Learning in cliff-walking environment
import math, os, time, sys
import numpy as np
import random
import gym
from agent import QLearningAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
from os import mkdir
import pickle as pkl
from analyse import analyse
##### END CODING HERE #####

# construct the environment
env = gym.make("CliffWalking-v0")
# get the size of action space 
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
random.seed(RANDOM_SEED) 
np.random.seed(RANDOM_SEED) 

##### START CODING HERE #####
# construct the intelligent agent.
df = 0.9
lr = 0.1
decay_rate = 0.9999
decay = lambda eps: eps * decay_rate  # eps-decay schema
agent = QLearningAgent(all_actions, decay, disc_factor=df, learning_rate=lr)

t = round(time.time() % 1e6)
dir_name = f"2000_q_conv_{str(df).split('.')[-1]}_{str(lr).split('.')[-1]}_{str(decay_rate).split('.')[-1]}_{t}"
mkdir(dir_name)
f_reward = open(f"{dir_name}/reward.txt", 'a')
f_eps = open(f"{dir_name}/eps.txt", 'a')

focuses = [(24,1), (30,1), (35,2),
        (24,0), (0,1), (6,1), (11,2)]
f_focuses = [open("{}/{}_{}.txt".format(dir_name, focus[0], focus[1]), 'a') for focus in focuses]

# start training
for episode in range(2000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    # render env. You can remove all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # print(f"#epi: {episode}  #iter: {iter}")
        # choose an action
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        # env.render()
        # update the episode reward
        episode_reward += r
        # print(f"{s} {a} {s_} {r} {isdone}")
        # agent learns from experience
        agent.learn(s, a, s_, r)
        s = s_

        if isdone:
            # time.sleep(0.1)
            break

    # print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)
    if episode % 200 == 0:
        with open(f"{dir_name}/Q_{episode}.pkl", 'wb') as f:
            pkl.dump(agent.Q, f)

    for (focus, f_focus) in zip(focuses, f_focuses):
        f_focus.write("{}, ".format(agent.Q.get(focus[0], {focus[1]: 0}).get(focus[1], 0)))

    f_reward.write(f"{episode_reward}, ")
    f_eps.write(f"{agent.epsilon}, ")
    f_reward.flush()
    f_eps.flush()
print('\ntraining over\n')

agent.plan(env, f"{dir_name}/path.txt", f"{dir_name}/Q_999.pkl")

f_reward.close()
f_eps.close()

# close the render window after training.
env.close()
analyse(dir_name)

##### END CODING HERE #####


