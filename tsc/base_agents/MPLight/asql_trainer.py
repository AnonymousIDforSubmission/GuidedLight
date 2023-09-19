#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import copy

from sumo_files.env.sim_env import TSCSimulator
from torch.utils.tensorboard import SummaryWriter
# import torch.multiprocessing as mp
from tsc.base_agents.MPLight.config import config
from tsc.utils import *
from tsc.base_agents.MPLight.DQN import DQNWorker
from tsc.base_agents.MPLight.mplight import NN_Model


def write_tensorboard(writer, loss, reward, step):
    writer.add_scalar('loss', loss, step)
    # writer.add_scalar('loss_critic', loss_critic, step)
    # writer.add_scalar('loss_entropy', loss_entropy, step)
    # writer.add_scalar('advantages', advantages, step)
    if reward:
        r = {}
        for tl, rs in reward.items():
            for k, v in rs.items():
                if k not in r:
                    r[k] = []
                r[k].append(v)
        total_reward = 0
        for k, v in r.items():
            writer.add_scalar('reward/{}'.format(k), np.mean(v), step)
            total_reward += np.mean(v)
            print('Step{}, reward_{}: {}'.format(step, k, np.mean(v)))
        writer.add_scalar('reward/all', total_reward, step)
    flush_writer(writer)


def train_worker(id, shared_NN, target_model, env_config, port, optimizer, rollout_counter, constants, device, lock):
    set_seed(id+2022)
    sumo_envs_num = len(env_config['sumocfg_files'])
    sumo_cfg = env_config['sumocfg_files'][id % sumo_envs_num]
    env_config = copy.deepcopy(env_config)
    env_config['sumocfg_file'] = sumo_cfg
    env = TSCSimulator(env_config, port)
    unava_phase_index = []
    for i in env.all_tls:
        unava_phase_index.append(env._crosses[i].unava_index)

    local_NN = NN_Model(8, 4, state_keys=env_config['state_key'], device=device).to(device)
    worker = DQNWorker(constants, device, env, shared_NN, target_model, local_NN, optimizer, id, lock)
    writer = SummaryWriter(comment=sumo_cfg+"/"+constants['model_save']['path'])
    while rollout_counter.get() < constants['episode']['num_train_rollouts'] + 1:
        loss, all_reward = worker.train_rollout(unava_phase_index)
        write_tensorboard(writer, loss, all_reward, rollout_counter.get())
        rollout_counter.increment()
        local_NN.decay_epsilon(rollout_counter.get())
        if rollout_counter.get() % constants['model_save']['frequency'] == 0:
            checkpoint(constants['model_save']['path'], shared_NN, optimizer, rollout_counter.get())

    # Kill connection to sumo server
    worker.env.terminate()
    print('...Training worker {} done'.format(id))


def train_AS_DQN(config, device):
    env_config = config['environment']
    shared_NN = NN_Model(8, 4, state_keys=env_config['state_key'], device=device).to(device)
    if not os.path.exists(config['model_save']['path']):
        os.mkdir(config['model_save']['path'])
    files = os.listdir(config['model_save']['path'])
    rollout_counter = Counter()
    if len(files) > 0:
        nums = []
        for i in files:
            nums.append(int(i.split("_")[-1].split(".")[0]))
        max_file = os.path.join(config['model_save']['path'], files[nums.index(max(nums))])
        shared_NN = load_checkpoint2(shared_NN, max_file)
        step = max(nums)
        rollout_counter.increment(step)
    target_model = NN_Model(8, 4, state_keys=env_config['state_key'], device=device).to(device)
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    target_model.share_memory()
    shared_NN.share_memory()
    optimizer = torch.optim.Adam(shared_NN.parameters(), config['DQN']['learning_rate'])
      # To keep track of all the rollouts amongst agents
    processes = []
    lock = mp.Lock()
    for i in range(config['parallel']['num_workers']):
        id = i
        port = env_config['port_start'] + i
        p = mp.Process(target=train_worker, args=(id, shared_NN, target_model, env_config, port,  optimizer,
                                                  rollout_counter, config, device, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')

    train_AS_DQN(config, device)
