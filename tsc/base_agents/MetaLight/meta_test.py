#!/usr/bin/env python3
# encoding: utf-8

import copy
import os
import sys
from tqdm.contrib.concurrent import process_map
from functools import partial
from eval_config import config
import pickle
from metalight_agent import MetaLightAgent
from sumo_files.env.sim_env import TSCSimulator
import gc
import objgraph


def get_reward(reward, all_tls):
    ans = []
    for i in all_tls:
        ans.append(sum(reward[i].values()))
    return ans

def worker(e_config, port, p):
    port = copy.deepcopy(port)
    port += p
    shared_dict_ = {}
    for k in e_config['reward_type'] + ['all']:
        shared_dict_[k] = []
    params = pickle.load(
        open(os.path.join(e_config['PATH_TO_MODEL'], 'params_{}.pkl'.format(p)), 'rb'))
    for i in range(5):
        try:
            gc.collect()
            model = MetaLightAgent(e_config)
            model.load_params([params])
            e_config['step_num'] = i + 1
            env = TSCSimulator(e_config, port)
            unava_phase_index = []
            for i in env.all_tls:
                unava_phase_index.append(env._crosses[i].unava_index)
            state = env.reset()
            next_state = state
            while True:
                next_state['unava'] = unava_phase_index
                next_state['tls'] = env.all_tls
                actions = model.choose_action([next_state], test=True)[0]
                tl_action_select = {}
                for tl_index in range(len(env.all_tls)):
                    tl_action_select[env.all_tls[tl_index]] = actions[tl_index]
                next_state, reward, done, _all_reward = env.step(tl_action_select)
                # reward = get_reward(reward, env.all_tls)
                if done:
                    all_reward = _all_reward
                    break
            for tl in all_reward.keys():
                all_reward[tl]['all'] = sum(all_reward[tl].values())
            for k in e_config['reward_type'] + ['all']:
                tmp = 0
                for tl in all_reward.keys():
                    tmp += all_reward[tl][k]
                shared_dict_[k].append(tmp / len(all_reward))
        except Exception as e:
            print(e)
            print(p, i)
        # shared_dict[p] = shared_dict_
    shared_dict = {}
    shared_dict[e_config['sumocfg_file'].split("/")[-1]] = {}
    for k, v in shared_dict_.items():
        shared_dict[e_config['sumocfg_file'].split("/")[-1]][k] = sum(v) / len(v)
        print("epoch:{}, {} Model Avg {}: {}".format(p, e_config['sumocfg_file'], k,
                                                     sum(v) / len(v)))
    with open(os.path.join(e_config['PATH_TO_MODEL'], "reward_record_{}_{}.pkl".format(e_config['sumocfg_file'].split("/")[-1], p)), "wb") as f:
        pickle.dump(shared_dict, f)
    # objgraph.show_most_common_types(limit=50)
    # for k, v in shared_dict_.items():
    #     shared_dict[p][k] = sum(v) / len(v)


def model_eval(e_config, port):
    process_map(partial(worker,  e_config, port), range(91, 100), max_workers=4)


if __name__ == '__main__':
    device = 'cpu'
    # config["environment"]['gui'] = True
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    all_reward_record = {}
    for i in range(len(config['sumocfg_files'])):
        e_config = copy.deepcopy(config)
        e_config['sumocfg_file'] = config['sumocfg_files'][i]
        e_config['sumocfg_files'] = [config['sumocfg_files'][i]]
        port = e_config['port_start']
        model_eval(e_config, port)

    data = {}
    for i in range(100):
        data[i] = {}
        for j in config['sumocfg_files']:
            name = j.split("/")[-1]
            with open(os.path.join(config['PATH_TO_MODEL'], "reward_record_{}_{}.pkl".format(name, i)),
                      "rb") as f:
                data_ = pickle.load(f)
            data[i][name] = data_

    with open(os.path.join(config['PATH_TO_MODEL'], "reward_record.pkl"), "wb") as f:
        pickle.dump(data, f)
