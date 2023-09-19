# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:29
 @Author  : 姜浩源
 @FileName: meta_train.py
 @Software: PyCharm
"""
from metalearner import MetaLearner
from metalight_agent import MetaLightAgent
from sampler import BatchSampler
from multiprocessing import Process
from config import config
import copy

import random
import numpy as np
import tensorflow as tf
import pickle
import shutil


def main(args):
    '''
        Perform meta-training for MAML and MetaLight

        Arguments:
            args: generated in utils.py:parse()
    '''

    # configuration: experiment, agent, traffic_env, path


    random.seed(args['SEED'])
    np.random.seed(args['SEED'])
    tf.set_random_seed(args['SEED'])

    # load or build initial model
    if not args.get('PRE_TRAIN', None):
        p = Process(target=build_init, args=(args, 1))
        p.start()
        p.join()
    else:
        if not os.path.exists(args['PATH_TO_MODEL']):
            os.makedirs(args['PATH_TO_MODEL'])
        shutil.copy(os.path.join('model', 'initial', 'common', args['PRE_TRAIN_MODEL_NAME'] + '.pkl'),
                    os.path.join(args['PATH_TO_MODEL'], 'params' + "_" + "init.pkl"))

    for batch_id in range(args['RUN_ROUND']):
        # meta batch size process
        process_list = []
        task_num = min(len(args['sumocfg_files']), 2)
        sample_task_traffic = random.sample(args['sumocfg_files'], task_num)
        if batch_id % 2 == 0:
            args['port_start'] += 2
        else:
            args['port_start'] -= 2
        p = Process(target=metalight_train,
                    args=(args,
                          sample_task_traffic, batch_id))
        p.start()
        p.join()

        ## update the epsilon
        decayed_epsilon = args["EPSILON"] * pow(args["EPSILON_DECAY"], batch_id)
        args["EPSILON"] = max(decayed_epsilon, args["MIN_EPSILON"])


def build_init(args, _):
    '''
        build initial model for maml and metalight

        Arguments:
            dic_agent_conf:         configuration of agent
            dic_traffic_env_conf:   configuration of traffic environment
            dic_path:               path of source files and output files
    '''
    args['sumocfg_files'] = args['sumocfg_files'][:1]
    policy = MetaLightAgent(args)
    params = policy.init_params()
    if not os.path.exists(args["PATH_TO_MODEL"]):
        os.makedirs(args["PATH_TO_MODEL"])
    pickle.dump(params, open(os.path.join(args['PATH_TO_MODEL'], 'params' + "_" + "init.pkl"), 'wb'))



def metalight_train(args, tasks, batch_id):
    '''
        metalight meta-train function

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            _dic_traffic_env_conf:  dict,   configuration of traffic environment
            _dic_path:              dict,   path of source files and output files
            tasks:                  list,   traffic files name in this round
            batch_id:               int,    round number
    '''
    args = copy.deepcopy(args)
    args['sumocfg_files'] = []
    for task in tasks:
        args['sumocfg_files'].append(task)
    sampler = BatchSampler(args)

    policy = MetaLightAgent(args)

    metalearner = MetaLearner(sampler, policy, args)

    if batch_id == 0:
        params = pickle.load(open(os.path.join(args['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        metalearner.meta_target_params = params

    else:
        params = pickle.load(open(os.path.join(args['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)), 'rb'))
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        period = args['PERIOD']
        target_id = int((batch_id - 1)/ period)
        meta_params = pickle.load(open(os.path.join(args['PATH_TO_MODEL'], 'params_%d.pkl' % (target_id * period)), 'rb'))
        meta_params = [meta_params] * len(policy.policy_inter)
        metalearner.meta_target_params = meta_params

    metalearner.sample_metalight(tasks, batch_id)


if __name__ == '__main__':
    import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    main(config)
