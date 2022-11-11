# -*- coding:utf-8 -*-
import math, os, time, sys
import numpy as np
import gym
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
import pickle as pkl
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

class SarsaAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, decay, disc_factor=1, learning_rate=0.1):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1
        self.decay = decay
        self.disc_factor = disc_factor
        self.learning_rate = learning_rate
        self.Q: dict[int, dict[int,float]] = {} 


    def choose_action(self, observation, exploration=1):
        """choose action with epsilon-greedy algorithm."""
        Q_s = self.Q.get(observation, {})  # Q_s[a] = Q(s,a)

        if exploration:
            prob = np.array([1-self.epsilon, self.epsilon])
            self.epsilon = self.decay(self.epsilon)
            # print(observation, Q_s)
            if not Q_s:
                return np.random.choice(self.all_actions)
            actions = [max(Q_s, key=Q_s.get), np.random.choice(self.all_actions)]
            return np.random.choice(np.array(actions), p=prob)
        
        return max(Q_s, key=Q_s.get) if Q_s else np.random.choice(self.all_actions)
    

    def learn(self, s, a, s_, r, a_):
        """learn from experience"""
        # time.sleep(0.5)
        # print("What I should learn? (ﾉ｀⊿´)ﾉ")
        if s not in self.Q:
            self.Q[s] = {a:0 for a in self.all_actions}
        if s_ not in self.Q:
            self.Q[s_] = {a:0 for a in self.all_actions}
        self.Q[s][a] += self.learning_rate * (
            r + self.disc_factor * self.Q[s_][a_] - self.Q[s][a])
        # print(self.Q)
        # return False
    

    def plan(self, env, path_file_path, table_file_path):
        """plan a path and output the path and Q-table to files"""
        with open(table_file_path, 'wb') as f_table:
            pkl.dump(self.Q, f_table)
        f_path = open(path_file_path, 'a')

        reward = 0
        # reset env
        s = env.reset()
        # render env. You can remove all render() to turn off the GUI to accelerate training.
        # env.render()
        # choose an action
        a = self.choose_action(s, exploration=0)
        # agent interacts with the environment
        while 1:
            s_, r, isdone, info = env.step(a)
            f_path.write(f'{s}, {a}, {s_}\n')
            f_path.flush()
            # env.render()
            # update the reward
            reward += r
            # print(f"{s} {a} {s_} {r} {isdone}")
            # choose an action
            a_ = self.choose_action(s_, exploration=0)
            # agent learns from experience
            # self.learn(s, a, s_, r, a_)
            s = s_
            a = a_
            if isdone:            
                f_path.close()
                return

    ##### END CODING HERE #####


class QLearningAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, decay, disc_factor=1, learning_rate=0.1):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1
        self.decay = decay
        self.disc_factor = disc_factor
        self.learning_rate = learning_rate
        self.Q: dict[int, dict[int,float]] = {} 


    def choose_action(self, observation, exploration=1):
        """choose action with epsilon-greedy algorithm."""
        Q_s = self.Q.get(observation, {})  # Q_s[a] = Q(s,a)

        if exploration:
            prob = np.array([1-self.epsilon, self.epsilon])
            self.epsilon = self.decay(self.epsilon)
            # print(observation, Q_s)
            if not Q_s:
                return np.random.choice(self.all_actions)
            actions = [max(Q_s, key=Q_s.get), np.random.choice(self.all_actions)]
            return np.random.choice(np.array(actions), p=prob)

        return max(Q_s, key=Q_s.get) if Q_s else np.random.choice(self.all_actions)

    
    def learn(self, s, a, s_, r):
        """learn from experience"""
        # time.sleep(0.5)
        # print("What I should learn? (ﾉ｀⊿´)ﾉ")
        if s not in self.Q:
            self.Q[s] = {a:0 for a in self.all_actions}
        nextQ = max(self.Q[s_].values()) if s_ in self.Q else 0
        self.Q[s][a] += self.learning_rate * (
            r + self.disc_factor * nextQ - self.Q[s][a])
        # print(self.Q)
        # return False

    
    def plan(self, env, path_file_path, table_file_path):
        """plan a path and output the path and Q-table to files"""
        with open(table_file_path, 'wb') as f_table:
            pkl.dump(self.Q, f_table)
        f_path = open(path_file_path, 'a')

        reward = 0
        # reset env
        s = env.reset()
        # render env. You can remove all render() to turn off the GUI to accelerate training.
        # env.render()
        # agent interacts with the environment
        while 1:
            # choose an action
            a = self.choose_action(s, exploration=0)
            s_, r, isdone, info = env.step(a)
            f_path.write(f'{s}, {a}, {s_}\n')
            f_path.flush()
            # env.render()
            # update the episode reward
            reward += r
            # print(f"{s} {a} {s_} {r} {isdone}")
            # agent learns from experience
            # self.learn(s, a, s_, r)
            s = s_
            if isdone:            
                f_path.close()
                return


    ##### END CODING HERE #####
