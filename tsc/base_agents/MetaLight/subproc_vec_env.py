# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 14:24
 @Author  : 姜浩源
 @FileName: subproc_vec_env.py.py
 @Software: PyCharm
"""
import numpy as np
import multiprocessing as mp
import copy
from sumo_files.env.sim_env import TSCSimulator
import multiprocessing


# class EnvWorker(mp.Process):
#     def __init__(self, remote, port, config, queue, lock):
#         super(EnvWorker, self).__init__()
#         self.remote = remote
#         self.queue = queue
#         self.lock = lock
#         self.task_id = None
#         self.done = False
#
#         self.env = TSCSimulator(config, port)
#
#     # def empty_step(self):
#     #     observation = [{'cur_phase': [0], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]
#     #     reward, done = [0.0], True
#     #     return observation, reward, done, []
#
#     def try_reset(self):
#         # with self.lock:
#         #     try:
#         #         self.task_id = self.queue.get(True)
#         #         self.done = (self.task_id is None)
#         #     except queue.Empty:
#         #         self.done = True
#         # if not self.done:
#         #     if "test_round" not in self.dic_path['PATH_TO_LOG']:
#         #         new_path_to_log = os.path.join(self.dic_path['PATH_TO_LOG'],
#         #                                                     'episode_%d' % (self.task_id))
#         #     else:
#         #         new_path_to_log = self.dic_path['PATH_TO_LOG']
#         #     self.env.modify_path_to_log(new_path_to_log)
#         #     if not os.path.exists(new_path_to_log):
#     #         os.makedirs(new_path_to_log)
#         state = self.env.reset()
#         #observation = (np.zeros(self.env.observation_space.shape,
#         #    dtype=np.float32) if self.done else self.env.reset())
#         return state
#         # else:
#         #     return False
#
#     def run(self):
#         while True:
#             command, data = self.remote.recv()
#             if command == 'step':
#                 observation, reward, done, info = self.env.step(data)
#                 if done:
#                     self.try_reset()
#                     #observation = self.try_reset()
#                 self.remote.send((observation, reward, done, self.task_id, info, self.done))
#             elif command == 'reset':
#                 observation = self.try_reset()
#                 self.remote.send((observation, self.task_id))
#             # elif command == 'reset_task':
#             #     self.env.unwrapped.reset_task(data)
#             #     self.remote.send(True)
#             elif command == 'close':
#                 self.remote.close()
#                 break
#             # elif command == 'get_spaces':
#             #     self.remote.send((self.env.observation_space,
#             #                      self.env.action_space))
#             # elif command == 'bulk_log':
#             #     self.env.bulk_log()
#             #     self.remote.send(True)
#             else:
#                 raise NotImplementedError()

# def _worker(config, i):
#     _config = copy.deepcopy(config)
#     _config['sumocfg_file'] = config['sumocfg_files'][i]
#     port = config['port_start'] + i
#     env = TSCSimulator(_config, port)
#     print(1)

def _worker(remote, parent_remote, config, i):
    parent_remote.close()
    _config = copy.deepcopy(config)
    _config['sumocfg_file'] = config['sumocfg_files'][i]
    port = config['port_start'] + i
    env = TSCSimulator(_config, port)
    unava_phase_index = []
    all_tls = env.all_tls
    for i in env.all_tls:
        unava_phase_index.append(env._crosses[i].unava_index)
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                actions = {}
                for i, tl in enumerate(all_tls):
                    actions[tl] = data[i]
                observation, reward, done, _all_reward = env.step(actions)
                # if done:
                #     # save final observation where user can get it, then reset
                #     observation = env.reset()
                observation['unava'] = unava_phase_index
                observation['tls'] = all_tls
                remote.send((observation, reward, done, _all_reward))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset()
                observation['unava'] = unava_phase_index
                observation['tls'] = all_tls
                remote.send(observation)
            elif cmd == 'close':
                env.terminate()
                remote.close()
                break
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break


class SubprocVecEnv():
    def __init__(self, num_workers, config):
        """
            Environment controller: single controller (agent) multiple environments
            Arguments:
                num_workers: number of environments (worker)
                queue:       process queue
        """
        self.lock = mp.Lock()

        self.processes = []

#
        self.remotes, self.work_remotes = zip(*[mp.Pipe(duplex=True) for _ in range(num_workers)])
        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            # args = (work_remote, remote, CloudpickleWrapper(env_fn))
            args = (work_remote, remote, config, i)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = mp.Process(target=_worker, args=args,
                                  daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.waiting = False
        self.closed = False

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, infos = zip(*results)

        return np.stack(observations), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        # observations = zip(*results)
        # return np.stack(observations)
        return results

    def reset_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self. waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True


if __name__ == '__main__':
    config = {
        "name": "test",
        "agent": "",
        'port_start': 15900,
        "sumocfg_files": [
            # "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
            #               "sumo_files/scenarios/nanshan/osm.sumocfg",
            "sumo_files/scenarios/sumo_wj3/rl_wj.sumocfg",
            "sumo_files/scenarios/sumo_wj3/rl_wj.sumocfg"
        ],
        "action_type": "select_phase",
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len'],
        'state_key': ['current_phase', 'stop_car_num']
    }
    # queue = mp.Queue()
    # num_workers = len(config['sumocfg_files'])
    # processes = []
    # for i in range(num_workers):
    #     args = (config, i)
    #     # daemon=True: if the main process crashes, we should not cause things to hang
    #     p = mp.Process(target=worker, args=args)  # pytype:disable=attribute-error
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    queue = mp.Queue()
    num_worker = len(config['sumocfg_files'])
    a = SubprocVecEnv(num_worker, config)
    print(a.reset())
    e1_tls = ['nt1',
 'nt10',
 'nt11',
 'nt12',
 'nt13',
 'nt14',
 'nt15',
 'nt16',
 'nt17',
 'nt18',
 'nt19',
 'nt2',
 'nt20',
 'nt21',
 'nt22',
 'nt23',
 'nt24',
 'nt25',
 'nt3',
 'nt4',
 'nt5',
 'nt6',
 'nt7',
 'nt8',
 'nt9']
    e2_tls = ['1169612816',
 '1236299368',
 '1605987626',
 '2132634162',
 '2184499023',
 '2184499026',
 '2184499119',
 '2184499120',
 '278660698',
 '7223736528',
 '830984681',
 'cluster_1116448404_5175418178_9128487758_9128487765',
 'cluster_1169603832_2425050827',
 'cluster_1236299390_2958627126',
 'cluster_2184499051_2184499060',
 'cluster_2184499054_2184499062',
 'cluster_2184499121_2402273641',
 'cluster_2290893768_2425050866',
 'cluster_2402273648_278376591',
 'cluster_2420580030_267602574',
 'cluster_2462495536_830984674',
 'cluster_2469773718_279073848_9197027640_9197027673',
 'cluster_4811663096_5815685483',
 'cluster_5175418461_9128520952',
 'cluster_5175418474_9128520950',
 'cluster_6901200204_830982935_9128520948_9128520949',
 'cluster_9044994673_9044994679_9044994680_9044994702_9044994704_9044994859_9044994860',
 'cluster_9105636781_9105636816_9105636824_9105636827']
    e3_tls = ['ftddj_frj', 'ftddj_wjj']
    print(a.step([{n:0 for n in e3_tls},
        {n:0 for n in e3_tls},
        # {n:0 for n in e3_tls}
    ]))
    a.close()
