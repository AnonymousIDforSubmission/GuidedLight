# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:47
 @Author  : 姜浩源
 @FileName: metalearner.py
 @Software: PyCharm
"""
import pickle
import os


class MetaLearner(object):
    def __init__(self, sampler, policy, args):
        """
            Meta-learner incorporates MAML and MetaLight and can update the meta model by
            different learning methods.
            Arguments:
                sampler:    sample trajectories and update model parameters
                policy:     frapplus_agent or metalight_agent
                ...
        """
        self.sampler = sampler
        self.policy = policy
        self.config = args
        self.meta_params = self.policy.save_params()
        self.meta_target_params = self.meta_params
        self.step_cnt = 0

    def sample_metalight(self, _tasks, batch_id):
        """
            Use MetaLight framework to samples trajectories before and after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        self.batch_id = batch_id
        tasks = []
        for task in _tasks:
            tasks.extend([task] * self.config['FAST_BATCH_SIZE'])
        self.sampler.reset_task(_tasks, self.config)
        meta_params = self.sampler.sample_metalight(self.policy, tasks, batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params)
        pickle.dump(meta_params, open(
           os.path.join(self.sampler.config['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

    def sample_meta_test(self, task, batch_id, old_episodes=None):
        """
            Perform meta-testing (only testing within one episode) or offline-training (in multiple episodes to let models well trained and obtrained pre-trained models).
            Arguments:
                old_episodes: episodes generated and kept in former batches, controlled by 'MULTI_EPISODES'
                ...
        """
        self.batch_id = batch_id
        tasks = [task] * self.config['FAST_BATCH_SIZE']
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')

        self.meta_params, self.meta_target_params, episodes = \
            self.sampler.sample_meta_test(self.policy, tasks[0], batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params, old_episodes=old_episodes)
        pickle.dump(self.meta_params, open(
            os.path.join(self.sampler.dic_path['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))
        return episodes
