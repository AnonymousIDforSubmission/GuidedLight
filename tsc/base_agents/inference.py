#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
from maxpressure import MAXPRESSURE
from inference_config import config
from sumo_files.env.sim_env import TSCSimulator
from tsc.utils import batch


PHASE_PAIRS = [
    [3, 7],
    [6, 7],
    [2, 3],
    [2, 6],
    [1, 5],
    [4, 5],
    [0, 1],
    [0, 4]
]


def eval(model):
    all_reward_list = {}
    for k in config['environment']['reward_type'] + ['all']:
        all_reward_list[k] = []
    for i in range(5):
        output_path = env_config['output_path_head'] + f'trial_{i}'
        os.makedirs(output_path, exist_ok=True)
    for i in range(5):
        env_config['step_num'] = i + 1
        env_config['p'] = 1
        env_config['output_path'] = env_config[
                                        'output_path_head'] + f'trial_{i}/'  # 不同的重复实验保存到不同的文件夹
        env = TSCSimulator(env_config, port)
        unava_phase_index = []
        for i in env.all_tls:
            unava_phase_index.append(env._crosses[i].unava_index)
        state = env.reset()
        next_state = state
        while True:
            state = batch(next_state, config['environment']['state_key'], env.all_tls)
            action = model.act(state, unava_phase_index)
            # action = get_action(prediction)
            tl_action_select = {}
            for tl_index in range(len(env.all_tls)):
                tl_action_select[env.all_tls[tl_index]] = \
                    (env._crosses[env.all_tls[tl_index]].green_phases)[action[tl_index]]
            next_state, reward, done, _all_reward = env.step(tl_action_select)
            # reward = get_reward(reward, env.all_tls)
            if done:
                all_reward = _all_reward
                break
        for tl in all_reward.keys():
            all_reward[tl]['all'] = sum(all_reward[tl].values())
        for k in config['environment']['reward_type'] + ['all']:
            tmp = 0
            for tl in all_reward.keys():
                tmp += all_reward[tl][k]
            all_reward_list[k].append(tmp / len(all_reward))

    for k, v in all_reward_list.items():
        print("{} Model Avg {}: {}".format(env_config['sumocfg_file'], k, sum(v) / len(v)))


if __name__ == '__main__':
    device = 'cpu'
    # config["environment"]['gui'] = True
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    env_config = config['environment']
    model = MAXPRESSURE({}, PHASE_PAIRS)
    for i in range(len(env_config['sumocfg_files'])):
        env_config['sumocfg_file'] = env_config['sumocfg_files'][i]
        port = env_config['port_start']
        eval(model)
