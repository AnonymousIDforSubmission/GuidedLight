# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:42
 @Author  : 姜浩源
 @FileName: metalight_agent.py
 @Software: PyCharm
"""
import numpy as np
from frapplus_agent import FRAPPlusAgent
import copy

class MetaLightAgent:
    def __init__(self, config):
        '''
            MetaLightAgent incorporates some FRAPPlusAgents. The number of FRAPPlusAgent is the same as the number the task number in one batch.
        '''
        self.config = config
        self.policy_inter = []
        for _ in config['sumocfg_files']:
            self.policy_inter.append(FRAPPlusAgent(config))
        self.group_size = self.config["FAST_BATCH_SIZE"]
        # else:
        #     for i in range(len(dic_traffic_env_conf)):
        #         self.policy_inter.append(FRAPPlusAgent(
        #             dic_agent_conf=dic_agent_conf,
        #             dic_traffic_env_conf=dic_traffic_env_conf[i],
        #             dic_path=dic_path[i])
        #         )
        #     self.group_size = self.dic_traffic_env_conf[0]["FAST_BATCH_SIZE"]

    def choose_action(self, observations, test=False):
        action_inter = []
        for i in range(int(len(observations) / self.group_size)):
            a = i * self.group_size
            b = (i + 1) * self.group_size
            #observs = [observations[j][i] for j in range(observations)]
            # print(self.policy_inter[i].choose_action(observations[a:b], test))
            action_inter.append(self.policy_inter[i].choose_action(observations[a:b], test))
            # print(action_inter)
        return action_inter

    def load_params(self, params):
        for i in range(len(self.policy_inter)):
            self.policy_inter[i].load_params(params[i])

    def fit(self, episodes, params, target_params):
        for i in range(len(self.policy_inter)):
            self.policy_inter[i].fit(episodes.episodes_inter[i], params=params[i], target_params=target_params[i])

    def update_params(self, episodes, params, lr_step, slice_index):
        new_params = []
        for i in range(len(self.policy_inter)):
            new_params.append(self.policy_inter[i].update_params(episodes.episodes_inter[i],
                                                                 params[i], lr_step, slice_index[i]))

        return new_params

    def init_params(self):
        return self.policy_inter[0].save_params()

    def save_params(self):
        params = []
        for policy in self.policy_inter:
            params.append(policy.save_params())
        return params

    def decay_epsilon(self, batch_id):
        for policy in self.policy_inter:
            policy.decay_epsilon(batch_id)

    def update_meta_params(self, episodes, slice_index, new_slice_index, _params):
        params = _params[0]

        tot_grads = dict(zip(params.keys(), [0] * len(params.keys())))
        for i in range(len(self.policy_inter)):
            grads = self.policy_inter[i].second_cal_grads(episodes.episodes_inter[i], slice_index[i], new_slice_index[i], params)

            for key in params.keys():
                tot_grads[key] += grads[key]

        for key in tot_grads.keys():
            tot_grads[key] = np.clip(tot_grads[key], -1 * self.config['CLIP_SIZE'],
                                  self.config['CLIP_SIZE'])

        new_params = dict(zip(params.keys(),
                               [params[key] - self.config["BETA"] * tot_grads[key] for key in params.keys()]))
        return [new_params] * len(_params)