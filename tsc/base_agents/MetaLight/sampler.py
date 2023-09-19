# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:09
 @Author  : 姜浩源
 @FileName: sampler.py
 @Software: PyCharm
"""
import multiprocessing as mp
from episode import BatchEpisodes, SeperateEpisode
import json
import os
import shutil
import random
import copy
import numpy as np
from subproc_vec_env import SubprocVecEnv


class BatchSampler(object):
    def __init__(self, config):
        """
            Sample trajectories in one episode by different methods
        """
        self.config = config
        self.task_path_map = {}
        self.task_traffic_env_map = {}


        self.queue = mp.Queue()
        self.envs = None
        self._task_id = 0

        self.step = 0
        self.target_step = 0
        self.lr_step = 0

        self.test_step = 0

    def sample_metalight(self, policy, tasks, batch_id, params=None, target_params=None, episodes=None):
        for i in range(len(tasks)):
            self.queue.put(i)
        for _ in range(len(tasks)):
            self.queue.put(None)

        size = int(len(tasks) / self.config["FAST_BATCH_SIZE"])
        episodes = SeperateEpisode(size, self.config["FAST_BATCH_SIZE"], self.config)

        observations = self.envs.reset()
        dones = [False]
        if params: # todo precise load parameter logic
            policy.load_params(params)

        old_params = None
        meta_update_period = 1
        meta_update = False

        while not all(dones):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            # actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, _all_rewards = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards)
            observations = new_observations

            # if update
            if self.step > self.config['UPDATE_START'] and self.step % self.config['UPDATE_PERIOD'] == 0:

                # if len(episodes) > self.config['MAX_MEMORY_LEN']:
                #     #TODO
                #     episodes.forget()

                old_params = params

                policy.fit(episodes, params=params, target_params=target_params)
                all_len = episodes.length()
                sample_size = [min(self.config['SAMPLE_SIZE'], i) for i in all_len]
                slice_index = [random.sample(range(all_len[i]), sample_size[i]) for i in range(len(sample_size))]
                params = policy.update_params(episodes, params=copy.deepcopy(params),
                                              lr_step=self.lr_step, slice_index=slice_index)
                policy.load_params(params)

                self.target_step += 1
                if self.target_step == self.config['UPDATE_Q_BAR_FREQ']:
                    target_params = params
                    self.target_step = 0

                # meta update
                if meta_update_period % self.config["META_UPDATE_PERIOD"] == 0:
                    policy.fit(episodes, params=params, target_params=target_params)
                    all_len = episodes.length()
                    sample_size = [min(self.config['SAMPLE_SIZE'], i) for i in all_len]
                    new_slice_index = [random.sample(range(all_len[i]), sample_size[i]) for i in
                                   range(len(sample_size))]
                    params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
                    policy.load_params(params)

                meta_update_period += 1

            self.step += 1
        print(_all_rewards)

        if not meta_update:
            policy.fit(episodes, params=params, target_params=target_params)
            all_len = episodes.length()
            sample_size = [min(self.config['SAMPLE_SIZE'], i) for i in all_len]
            new_slice_index = [random.sample(range(all_len[i]), sample_size[i]) for i in
                               range(len(sample_size))]
            params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
            policy.load_params(params)

            meta_update_period += 1

        policy.decay_epsilon(batch_id)
        return params[0]

    def reset_task(self, tasks, config):
        config = copy.deepcopy(config)
        config['sumocfg_files'] = tasks
        self.envs = SubprocVecEnv(len(tasks), config)

