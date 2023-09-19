#!/usr/bin/env python3
# encoding: utf-8

from tsc.PPO import Worker
import torch
import numpy as np
from copy import deepcopy
from tsc.utils import tensor, random_sample, ensure_shared_grads, batch


def copy_to_target(model, target):
    target.load_state_dict(model.state_dict())
    target.eval()
    for param in target.parameters():
        target.requires_grad = False


def moving_copy_to_target(model, target, tau=0.95):
    for p, tp in zip(model.parameters(), target.parameters()):
        tp.data = (1 - tau) * p.data + tau * tp.data


class DQNWorker(Worker):
    def __init__(self, constants, device, env, shared_model, target_model, local_model, optimizer, id, lock, num_agnet=None,
                 dont_reset=False):
        super(DQNWorker, self).__init__(constants, device, env, id)
        self.target_NN = target_model
        self.shared_NN = shared_model
        if not dont_reset:  # for the vis agent script this messes things up
            self.state = self.env.reset()
        self.ep_step = 0
        self.opt = optimizer
        self.lock = lock

        self.num_agents = len(env.all_tls)
        self.all_tls = self.env.all_tls
        self.state_keys = ['mask'] + constants['environment']['state_key']
        self.target_NN.eval()
        self.local_model = local_model
        self.local_model.train()
        self.crit = torch.nn.MSELoss()

    def _copy_shared_model_to_local(self):
        self.local_model.load_state_dict(self.shared_NN.state_dict())

    def _get_prediction(self, states, unava_phase_index):
        return self.local_model(states, unava_phase_index)

    def _get_action(self, q_pred, unava_phase_index, test=False):
        return self.local_model.choose_action(q_pred, unava_phase_index, test)

    def _get_target_prediction(self, states, unava_phase_index, action):
        return self.target_NN(states, unava_phase_index)[range(len(unava_phase_index)), action]

    def train_rollout(self, unava_phase_index=None):
        all_reward = None
        rollout_amt = 0
        state = deepcopy(self.state)
        while rollout_amt < self.constants['episode']['rollout_length']:
            with self.lock:
                if rollout_amt % 16:
                    self._copy_shared_model_to_local()

            self.local_model.zero_grad()
            state = batch(state, self.constants['environment']['state_key'], self.all_tls)
            q_pred = self._get_prediction(state, unava_phase_index)
            action = self._get_action(q_pred.detach(), unava_phase_index)
            tl_action_select = {}
            for tl_index in range(len(self.all_tls)):
                tl_action_select[self.all_tls[tl_index]] = \
                    (self.env._crosses[self.all_tls[tl_index]].green_phases)[action[tl_index]]
            next_state, reward, done, _all_reward = self.env.step(tl_action_select)
            reward = self.get_reward(reward)

            next_state_ = batch(next_state, self.constants['environment']['state_key'], self.all_tls)
            q_next_pred = self._get_prediction(next_state_, unava_phase_index)
            max_next_action = self._get_action(q_next_pred.detach(), unava_phase_index, True)
            q_next_target_pred = self._get_target_prediction(next_state_, unava_phase_index, max_next_action)
            dw = 0 if done else 1
            y_target = torch.FloatTensor(reward) + self.constants['DQN']['discount'] * dw * q_next_target_pred.detach()
            loss = self.crit(q_pred[range(len(unava_phase_index)), action], y_target)
            loss.backward()

            with self.lock:
                moving_copy_to_target(self.shared_NN, self.target_NN, tau=0.95)

            rollout_amt += 1
            if done:
                all_reward = _all_reward
                next_state = self.env.reset()
            state = deepcopy(next_state)

        torch.nn.utils.clip_grad_norm(self.local_model.parameters(), 10.)
        with self.lock:
            self.opt.zero_grad()
            ensure_shared_grads(self.local_model, self.shared_NN)
            self.opt.step()

        return loss, all_reward