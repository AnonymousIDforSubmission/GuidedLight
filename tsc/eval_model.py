#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
from eval_config import config
from sumo_files.env.sim_env import TSCSimulator
# from model import NN_Model
import numpy as np
import pickle
from utils import *
from model import NN_Model
from utils import load_checkpoint2
import torch


def get_action(prediction):
    return prediction['a'].cpu().detach().numpy()


def get_reward(reward, all_tls):
    ans = []
    for i in all_tls:
        ans.append(sum(reward[i].values()))
    return ans


def model_eval():
    all_reward_list = {}
    for k in config['environment']['reward_type'] + ['all']:
        all_reward_list[k] = []

    spe_model = config['model_save']['spe_path']
    #判断当前模型是属于所有checkpoints里第几个check
    p = int(spe_model.split('_')[1][:-3]) // config['model_save']['frequency']
    for i in range(1):#重复模拟的次数
        lstm_state = (
            torch.zeros(1, 10, 128).to(device),
            torch.zeros(1, 10, 128).to(device),
        )
        model = NN_Model(12, 4, state_keys=env_config['state_key'], device=device)
        # model = NN_Model(9, 4, state_keys=env_config['state_key'], device=device)
        model = load_checkpoint2(model, config['model_save']['path'] + spe_model) #读取指定的模型文件
        model.eval()
        # model = load_checkpoint(model, config['model_save']['path'])
        env_config['step_num'] = i + 1 #使用对应checkpoint所使用的route文件
        env_config['p'] = 2 * p + 1
        env_config['output_path'] = env_config['output_path_head'] + f'trial_{i}/' #不同的重复实验保存到不同的文件夹
        if not os.path.exists(env_config['output_path']):
            os.makedirs(env_config['output_path'])
        env = TSCSimulator(env_config, port)
        state = env.reset()
        unava_phase_index = []
        for j in env.all_tls:
            unava_phase_index.append(env._crosses[j].unava_index)
        tl_action = []
        tl_phase_duration = []
        tl_state = []
        tl_reward = []
        while True:
            tmp_state = {}
            for tl_index in range(len(env.all_tls)):
                tmp_state[env.all_tls[tl_index]] = state[env.all_tls[tl_index]]
            tl_state.append(tmp_state)
            state = batch(state, config['environment']['state_key'], env.all_tls)
            prediction, lstm_state, _ = model(state, lstm_state, unava_phase_index, eval=True)
            # action = get_action(prediction)

            action = prediction['a']
            # get phase duration

            for tl_index in range(len(env.all_tls)):
                tl_phase_duration.append({env.all_tls[tl_index]:
                    (env._crosses[env.all_tls[tl_index]].getCurrentPhase())})
                # tl_action.append({env.all_tls[tl_index]: action[tl_index]})
                tl_action.append({env.all_tls[tl_index]: [action[i][tl_index] for i in range(4)]})

            tl_action_select = {}
            # for tl_index in range(len(env.all_tls)):
            #     tl_action_select[env.all_tls[tl_index]] = action[tl_index]
            for tl_index in range(len(env.all_tls)):
                tl_action_select[env.all_tls[tl_index]] = []
                for index in range(4):
                    tl_action_select[env.all_tls[tl_index]].append(action[index][tl_index])

            state, reward, done, _all_reward = env.step(tl_action_select)
            tl_reward.append(reward)
            reward = get_reward(reward, env.all_tls)
            if done:
                all_reward = _all_reward
                with open(os.path.join(config['model_save']['path'],
                                       "plt_{}_{}.pkl".format(
                                               env_config['sumocfg_file'].split("/")[-1], p)),
                          "wb") as f:
                    pickle.dump({"tl_phase_duration": tl_phase_duration,
                                 "tl_state":tl_state,
                                 "tl_reward": tl_reward,
                                 "tl_action": tl_action},
                                f)
                break
        for tl in all_reward.keys():
            all_reward[tl]['all'] = sum(all_reward[tl].values())
        for k in config['environment']['reward_type']+['all']:
            tmp = 0
            for tl in all_reward.keys():
                tmp += all_reward[tl][k]
            all_reward_list[k].append(tmp/len(all_reward))

    for k, v in all_reward_list.items():
        print("{} Model Avg {}: {}".format(env_config['sumocfg_file'], k, sum(v)/len(v)))


def default_eval():
    # time = [25, 25, 25, 25] in intersection
    env_config['output_path'] = env_config['output_path_head'] + f'trial_0/'
    if not os.path.exists(env_config['output_path']):
        os.makedirs(env_config['output_path'])
    env = TSCSimulator(env_config, port, not_default=False)
    state = env.reset_default()
    tl_action = []
    tl_phase_duration = []
    tl_state = []
    tl_reward = []
    while True:
        tmp_state = {}
        for tl_index in range(len(env.all_tls)):
            tmp_state[env.all_tls[tl_index]] = state[env.all_tls[tl_index]]
        tl_state.append(tmp_state)
        for tl_index in range(len(env.all_tls)):
            tl_phase_duration.append({env.all_tls[tl_index]:
                                          (env._crosses[env.all_tls[tl_index]].getCurrentPhase())})
            tl_action.append({env.all_tls[tl_index]: [0,0,0,0]})
        state, reward, done, _all_reward = env.default_step()
        tl_reward.append(reward)
        if done:
            all_reward = _all_reward
            with open(os.path.join(config['model_save']['path'], "plt_tsc_tmp.pkl"), "wb") as f:
                pickle.dump({"tl_phase_duration": tl_phase_duration,
                             "tl_state":tl_state,
                             "tl_reward": tl_reward,
                             "tl_action": tl_action},
                            f)
            break
    for tl in all_reward.keys():
        all_reward[tl]['all'] = sum(all_reward[tl].values())
    for k in config['environment']['reward_type']+["all"]:
        tmp = 0
        for tl in all_reward.keys():
            tmp += all_reward[tl][k]
        print("{} Default Avg {}: {}".format(env_config['sumocfg_file'], k, tmp/len(all_reward)))


if __name__ == '__main__':
    device = 'cpu'
    # config["environment"]['gui'] = True
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    env_config = config['environment']
    for i in range(len(env_config['sumocfg_files'])):
        env_config['sumocfg_file'] = env_config['sumocfg_files'][i]
        port = env_config['port_start']

        model_eval() # eval model
        # default_eval()  # eval FTC
