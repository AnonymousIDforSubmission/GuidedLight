#!/usr/bin/env python3
# encoding: utf-8

import os
import sys

import torch

from config import config
from a3c_trainer import train_A3C_PPO


if config['environment'].get('is_libsumo', True):
    os.environ['LIBSUMO_AS_TRACI'] = '1'


def run(num_experiments=1):

    for exp in range(num_experiments):
        print(' --- Running experiment {} / {} --- '.format(exp + 1, num_experiments))
        run_experiment(config)


def run_experiment(constants):
    train_A3C_PPO(constants, device)


if __name__ == '__main__':
    # we need to import python modules from the $SUMO_HOME/tools directory
    torch.multiprocessing.set_start_method('spawn')
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    device = torch.device('cpu')

    run(num_experiments=1)
