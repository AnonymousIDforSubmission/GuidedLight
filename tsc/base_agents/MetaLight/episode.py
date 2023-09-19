# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:16
 @Author  : 姜浩源
 @FileName: episode.py
 @Software: PyCharm
"""
import numpy as np
import copy
import  config

class BatchEpisodes(object):
    def __init__(self, config, old_episodes=None):
        self.config = config

        self.total_samples = []

        self._observations = None
        self._actions = None
        self._rewards = None
        self._returns = None
        self._mask = None
        self.tot_x = []
        self.tot_next_x = []
        if old_episodes:
            self.total_samples = self.total_samples + old_episodes.total_samples
            self.tot_x = self.tot_x + old_episodes.tot_x
            self.tot_next_x = self.tot_next_x + old_episodes.tot_next_x

        self.last_x = []
        self.last_next_x = []
        self.current_x = []
        self.current_next_x = []

    def append(self, observations, actions, new_observations, rewards):
        self.last_x = self.current_x
        self.last_next_x = self.current_next_x
        self.current_x = []
        self.current_next_x = []

        for observation, action, new_observation, reward in zip(
                observations, actions, new_observations, rewards):
            # if batch_id is None:
            #     continue
            tls = observation['tls']
            unava = observation['unava']
            # print(observation, action, reward)
            for i, tl in enumerate(tls):
                self.total_samples.append([[observation[tl], unava[i]], action[i], new_observation[tl], reward[tl]])
                self.tot_x.append([observation[tl], unava[i]])
                self.current_x.append([observation[tl], unava[i]])

                self.tot_next_x.append([new_observation[tl], unava[i]])
                self.current_next_x.append([new_observation[tl], unava[i]])

    def get_x(self):
        return np.reshape(np.array(self.tot_x), (len(self.tot_x), -1))

    def get_next_x(self):
        return np.reshape(np.array(self.tot_next_x), (len(self.tot_next_x), -1))

    def forget(self):
        self.total_samples = self.total_samples[-1 * self.config['MAX_MEMORY_LEN'] : ]
        self.tot_x = self.tot_x[-1 * self.config['MAX_MEMORY_LEN'] : ]
        self.tot_next_x = self.tot_next_x[-1 * self.config['MAX_MEMORY_LEN']:]

    def prepare_y(self, q_values):
        self.tot_y = q_values

    def get_y(self):
        return self.tot_y

    def __len__(self):
        return len(self.total_samples)


class SeperateEpisode:
    def __init__(self, size, group_size, config, old_episodes=None):
        self.episodes_inter = []
        for _ in range(size):
            self.episodes_inter.append(BatchEpisodes(
                config=config, old_episodes=old_episodes)
            )
        self.num_group = size
        self.group_size = group_size

    def append(self, observations, actions, new_observations, rewards):

        for i in range(int(len(observations) / self.group_size)):
            a = i * self.group_size
            b = (i + 1) * self.group_size
            self.episodes_inter[i].append(observations[a : b], actions[a : b],
                                          new_observations[a : b], rewards[a : b])
        #for i in range(len(self.episodes_inter)):
        #    self.episodes_inter[i].append(observations[:, i], actions[:, i], new_observations[:, i], rewards[:, i], batch_ids)

    def forget(self, memory_len):
        for i in range(len(self.episodes_inter)):
            self.episodes_inter[i].forget(memory_len)

    def length(self):
        return [len(self.episodes_inter[i].total_samples) for i in range(len(self.episodes_inter))]

