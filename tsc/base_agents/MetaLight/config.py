# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:34
 @Author  : 姜浩源
 @FileName: config.py
 @Software: PyCharm
"""
import time
import os
_time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
name = 'metalight'
sumocfg_files = [
    "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
    "sumo_files/scenarios/large_grid2/exp_1.sumocfg",
    "sumo_files/scenarios/nanshan/osm.sumocfg",
    # "sumo_files/scenarios/sumo_wj3/rl_wj.sumocfg",
    "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",

    "sumo_files/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg",
    "sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg",
    "sumo_files/scenarios/resco_envs/grid4x4/grid4x4.sumocfg",
    "sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg"
]
config = {
    "FAST_BATCH_SIZE": 1,
    'SEED': 2022,
    'PATH_TO_MODEL': os.path.join('metalight', name+_time),
    'RUN_ROUND': 100,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LEARNING_RATE": 0.001,
    "ALPHA": 0.001,
    "MIN_ALPHA": 0.001,
    "ALPHA_DECAY_RATE": 0.95,
    "ALPHA_DECAY_STEP": 100,
    "BETA": 0.1,
    "LR_DECAY": 1,
    "MIN_LR": 0.0001,
    "SAMPLE_SIZE": 32,
    'UPDATE_START': 10,
    'PERIOD': 5,
    'ACTIVATION_FN': 'leaky_relu',
    'GRADIENT_CLIP': True,
    'CLIP_SIZE': 1,
    'OPTIMIZER': 'sgd',
    'UPDATE_PERIOD': 10,
    "TEST_PERIOD": 50,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 12000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "LOSS_FUNCTION": "mean_squared_error",
    "EARLY_STOP_LOSS": "val_loss",
    "DROPOUT_RATE": 0,
    "MORE_EXPLORATION": False,
    'NORM': None,
    "REWARD_NORM": False,
    "INPUT_NORM": False,
    'NUM_UPDATES' : 1,
    'NUM_GRADIENT_STEP': 1,
    "META_UPDATE_PERIOD": 10,

    'port_start': 15900,
    "sumocfg_files": sumocfg_files,
    "action_type": "select_phase",
    "gui": False,
    "yellow_duration": 5,
    "iter_duration": 10,
    "episode_length_time": 3600,
    'reward_type': ['queue_len'],
    'state_key': ['current_phase', 'stop_car_num'],
    'TRAFFIC_IN_TASKS': len(sumocfg_files)
}