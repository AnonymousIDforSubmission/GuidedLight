
import time
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from tsc.utils import Storage, tensor, seq_sample, ensure_shared_grads, batch

# scats
# from lightflow.service.minute_single_adaptive import MinuteSingleNodeAdaptive
# from lightflow.utils.enums import Direction, Movement
import json


def plan_update_mask(plan, direction, movement):
    for stage in plan['Stages']:
        for phase in stage['Phase']:
            if phase['Direction'] == int(direction) and phase['Movement'] == int(movement):
                stage['Phase'].remove(phase)
        if len(stage['Phase']) == 0:
            plan['Stages'].remove(stage)
    return plan

# below for scats

# def sdk_batch(input_data_adaptives):
#     '''
#     调用信控SDK 并转换为RL格式 example: [[phase1 duration, phase2 duration, phase3 duration, phase4 duration], .....]
#     '''
#     result = []
#     for input_data_adaptive in input_data_adaptives:
#         algo_res = MinuteSingleNodeAdaptive(input_data_adaptive,
#                                             **input_data_adaptive['AlgoConfig'])
#         # from lightflow.evaluate import evaluate_task
#         # print(evaluate_task(input_data_adaptive,algo_res,'single_adaptive'))
#
#         # 转换为RL格式
#         directions = [[int(Direction.east), int(Direction.west)],
#                       [int(Direction.south), int(Direction.north)]]
#         movements = [int(Movement.straight), int(Movement.left)]
#         tmp_result = []
#         for direction in directions:
#             for movement in movements:
#                 duration = 0
#                 for stage in algo_res['Stages']:
#                     for phase in stage['Phase']:
#                         if phase['Direction'] in direction and phase['Movement'] == movement:
#                             duration = max(duration, stage['PhasestageTime'])
#                 tmp_result.append(duration)
#         result.append(tmp_result)
#     return result


# def get_curr_state(index, batch_state):
#     state = {}
#     for key in batch_state.keys():
#         if isinstance(batch_state[key][0], torch.Tensor):
#             state[key] = batch_state[key][index].numpy()
#         else:
#             state[key] = batch_state[key][index]
#     return state

# def sdk_pre_batch(state):
#     '''
#     输入至lightflow预处理与数据转换
#     '''
#     with open('single_adaptive_high_052_20230330_070000.json', 'r+') as f:
#         input_data_adaptive = json.load(f)
#     directions = [Direction.north, Direction.west, Direction.south, Direction.east]
#     movements = [Movement.left, Movement.straight]
#     result = []
#
#     for index in range(state['flow'].shape[0]):
#         curr_state = get_curr_state(index, state)
#         index = 0
#         direction_flows = []
#         input_data_tmp = deepcopy(input_data_adaptive)
#         flow = input_data_tmp['TrafficVolume']['DirectionFlow'][0]
#         lane_list = deepcopy(input_data_tmp['Topology']['lane_list'])
#         for direction in directions:
#             for movement in movements:
#                 tmp_flow = deepcopy(flow)
#                 tmp_flow['Direction'] = str(int(direction))
#                 tmp_flow['Movement'] = str(int(movement))
#                 tmp_flow['Volume'] = curr_state['flow'][index]
#                 direction_flows.append(tmp_flow)
#                 # mask
#                 if curr_state['mask'][index] != 0:
#                     # delete from lane_list
#                     for lane in lane_list:
#                         if lane['Direction'] == str(int(direction)) and lane['Movement'] == str(int(movement)):
#                             lane_list.remove(lane)
#                     # delete from plan
#                     plan_update_mask(input_data_tmp['OriginalPlan'], direction, movement)
#                     plan_update_mask(input_data_tmp['PreviousPlan'], direction, movement)
#                 index += 1
#         input_data_tmp['TrafficVolume']['DirectionFlow'] = direction_flows
#         input_data_tmp['Topology']['lane_list'] = lane_list
#         # input_data_adaptive['PreviousPlan']['Cycle'] = 60
#         result.append(input_data_tmp)
#     return result


# def sdk_pre(state):
#     '''
#     输入至lightflow预处理与数据转换
#     '''
#     with open('single_adaptive_high_052_20230330_070000.json', 'r+') as f:
#         input_data_adaptive = json.load(f)
#     directions = [Direction.north, Direction.west, Direction.south, Direction.east]
#     movements = [Movement.left, Movement.straight]
#     result = {}
#     for cross_id in state.keys():
#         index = 0
#         direction_flows = []
#         input_data_tmp = deepcopy(input_data_adaptive)
#         flow = input_data_tmp['TrafficVolume']['DirectionFlow'][0]
#         lane_list = deepcopy(input_data_tmp['Topology']['lane_list'])
#         for direction in directions:
#             for movement in movements:
#                 tmp_flow = deepcopy(flow)
#                 tmp_flow['Direction'] = str(int(direction))
#                 tmp_flow['Movement'] = str(int(movement))
#                 tmp_flow['Volume'] = state[cross_id]['flow'][index]
#                 direction_flows.append(tmp_flow)
#                 # mask
#                 if state[cross_id]['mask'][index] != 0:
#                     # delete from lane_list
#                     for lane in lane_list:
#                         if lane['Direction'] == str(int(direction)) and lane['Movement'] == str(
#                                 int(movement)):
#                             lane_list.remove(lane)
#                     # delete from plan
#                     plan_update_mask(input_data_tmp['OriginalPlan'], direction, movement)
#                     plan_update_mask(input_data_tmp['PreviousPlan'], direction, movement)
#                 index += 1
#         input_data_tmp['TrafficVolume']['DirectionFlow'] = direction_flows
#         input_data_tmp['Topology']['lane_list'] = lane_list
#         # input_data_adaptive['PreviousPlan']['Cycle'] = 60
#         result[cross_id] = input_data_tmp
#     return result


# def sdk(input_data_adaptive_dict):
#     '''
#     调用信控SDK 并转换为RL格式 example: [[phase1 duration, phase2 duration, phase3 duration, phase4 duration], .....]
#     '''
#     result = {}
#     for cross_id in input_data_adaptive_dict.keys():
#         input_data_adaptive = input_data_adaptive_dict[cross_id]
#         algo_res = MinuteSingleNodeAdaptive(input_data_adaptive,
#                                             **input_data_adaptive['AlgoConfig'])
#         # from lightflow.evaluate import evaluate_task
#         # print(evaluate_task(input_data_adaptive,algo_res,'single_adaptive'))
#
#         # 转换为RL格式
#         directions = [[int(Direction.east), int(Direction.west)],
#                       [int(Direction.south), int(Direction.north)]]
#         movements = [int(Movement.straight), int(Movement.left)]
#         result[cross_id] = []
#         for direction in directions:
#             for movement in movements:
#                 duration = 0
#                 for stage in algo_res['Stages']:
#                     for phase in stage['Phase']:
#                         if phase['Direction'] in direction and phase['Movement'] == movement:
#                             duration = max(duration, stage['PhasestageTime'])
#                 result[cross_id].append(duration)
#
#     return result


def sdk_action2sim(action):
    action = np.array(action)
    action = np.where(action==0, 0.1, action)
    return action


def get_sdk_label(sampled_states, down_):
    current_duration = sampled_states['duration']
    flow = sampled_states['flow']
    # below is linear
    we_s = flow[:, 3] + flow[:, 7]
    we_l = flow[:, 2] + flow[:, 6]
    ns_s = flow[:, 1] + flow[:, 5]
    ns_l = flow[:, 0] + flow[:, 4]
    flows_ = torch.cat([we_s.unsqueeze(-1), we_l.unsqueeze(-1), ns_s.unsqueeze(-1), ns_l.unsqueeze(-1)],dim=-1)
    target_duration = flows_ * 0.35

    # below is scats
    # state = sdk_pre_batch(sampled_states)
    # action = sdk_batch(state)
    # target_duration = sdk_action2sim(action)
    # target_duration = np.clip(target_duration, down_, 90)

    mask = 1 - sampled_states['masked_phase']
    label_up = torch.zeros_like(current_duration).int()
    label_up += 2
    label_up[np.where((current_duration.numpy() + 5) < target_duration)] = 1
    label_up[np.where((current_duration.numpy() - 5) > target_duration)] = 0
    return label_up, mask


class Worker:
    def __init__(self, constants, device, env, id):
        self.constants = constants
        self.device = device
        self.env = env
        self.id = id

    def _reset(self):
        raise NotImplementedError

    def _get_prediction(self, states, actions=None, ep_step=None):
        raise NotImplementedError

    def _get_action(self, prediction):
        raise NotImplementedError

    def _copy_shared_model_to_local(self):
        raise NotImplementedError

    def get_reward(self, reward):
        ans = []
        for i in self.all_tls:
            ans.append(sum(reward[i].values()))
        return ans


# Code adapted from: Shangtong Zhang (https://github.com/ShangtongZhang)
class PPOWorker(Worker):
    def __init__(self, constants, device, env, shared_NN, local_NN,
                 optimizer, id, dont_reset=False):
        super(PPOWorker, self).__init__(constants, device, env, id)
        self.NN = local_NN
        self.shared_NN = shared_NN
        if not dont_reset:  # for the vis agent script this messes things up
            self.state = self.env.reset()
            self.unava_phase_index = []
            for i in env.all_tls:
                self.unava_phase_index.append(env._crosses[i].unava_index)
        self.ep_step = 0
        self.opt = optimizer
        self.num_agents = len(env.all_tls)
        self.all_tls = self.env.all_tls
        self.state_keys = ['mask', 'phase_index', 'masked_phase'] + constants['environment']['state_key']
        self.next_lstm_state = (
            torch.zeros(1, 10, 128).to(device),
            torch.zeros(1, 10, 128).to(device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
        self.NN.eval()
        self.alpha = 0.5

    def _get_prediction(self, states, next_lstm_state, unava_phase_index, actions=None, ep_step=None):
        return self.NN(states, next_lstm_state, unava_phase_index, actions)

    def _get_action(self, prediction):
        return prediction['a'].cpu().numpy()

    def _copy_shared_model_to_local(self):
        self.NN.load_state_dict(self.shared_NN.state_dict())

    def _stack(self, val):
        assert not isinstance(val, list)
        return np.stack([val] * self.num_agents)

    def train_rollout(self, total_step):
        storage = Storage(self.constants['episode']['rollout_length'])
        state = deepcopy(self.state)
        step_times = []
        all_reward = None
        self._copy_shared_model_to_local()
        lstm_state = deepcopy(self.next_lstm_state)
        done = False
        # initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        while not done:
            start_step_time = time.time()
            state = batch(state, self.constants['environment']['state_key'], self.all_tls)
            prediction, next_lstm_state, _ = self._get_prediction(state, lstm_state, self.unava_phase_index)
            action = prediction['a']
            # action = self._get_action(prediction)

            tl_action_select = {}
            # for tl_index in range(len(self.all_tls)):
            #     tl_action_select[self.all_tls[tl_index]] = action[tl_index]
            for tl_index in range(len(self.all_tls)):
                tl_action_select[self.all_tls[tl_index]] = []
                for index in range(4):
                    tl_action_select[self.all_tls[tl_index]].append(action[index][tl_index])
            next_state, reward, done, _all_reward = self.env.step(tl_action_select)
            reward = self.get_reward(reward)

            self.ep_step += 1

            # This is to stabilize learning, since the first some amt of states wont represent the env very well
            # since it will be more empty than normal

            storage.add(prediction)
            storage.add({'r': tensor(reward, self.device).unsqueeze(-1),
                         'm': tensor(self._stack(1 - done), self.device).unsqueeze(-1)})
            for k in self.state_keys:
                storage.add({k: tensor(state[k], self.device)})
            # storage.add({'lstm_state': tensor(torch.cat(lstm_state).permute(1,0,2), self.device)})
            storage.extend({"unava_index": self.unava_phase_index})
            state = next_state
            lstm_state = next_lstm_state

            end_step_time = time.time()
            step_times.append(end_step_time - start_step_time)
        self.next_lstm_state = (
                    torch.zeros(1, 10, 128),
                    torch.zeros(1, 10, 128),
                )
        state = batch(state, self.constants['environment']['state_key'], self.all_tls)
        with torch.no_grad():
            prediction, _, _ = self._get_prediction(state, lstm_state, self.unava_phase_index)
        storage.add(prediction)
        storage.placeholder()

        self.state = deepcopy(self.env.reset())
        self.unava_phase_index = []
        for i in self.env.all_tls:
            self.unava_phase_index.append(self.env._crosses[i].unava_index)
        self.ep_step = 0
        all_reward = _all_reward

        advantages = tensor(np.zeros((self.num_agents, 1)), self.device)
        returns = prediction['v'].detach()
        for i in reversed(range(self.constants['episode']['rollout_length'])):
            # Disc. Return
            returns = storage.r[i] + self.constants['ppo']['discount'] * storage.m[i] * returns
            # GAE
            td_error = storage.r[i] + self.constants['ppo']['discount'] * storage.m[i] * \
                       storage.v[i + 1] - storage.v[i]
            advantages = advantages * self.constants['ppo']['gae_tau'] * \
                         self.constants['ppo']['discount'] * storage.m[i] + td_error
            # tmp = {'adv': advantages.detach(),
            #        'ret': returns.detach()}
            # storage.insert(tmp)
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()


        actions_tmp = storage.a[:self.constants['episode']['rollout_length']]
        actions = []
        for g in range(4):
            tmp_a = []
            for step_i in range(len(actions_tmp)):
                for ii in range(len(actions_tmp[step_i][g])):
                    tmp_a.append(actions_tmp[step_i][g][ii])
            actions.append(tmp_a)
        log_probs_old, returns, advantages = storage.cat(['log_pi_a', 'ret', 'adv'])
        # actions, log_probs_old, returns, advantages = storage.cat(['a', 'log_pi_a', 'ret', 'adv'])
        states = storage.cat([k for k in self.state_keys])
        unava_phase_indexs = storage.unava_index
        states_ = {}
        for k, v in enumerate(states):
            states_[self.state_keys[k]] = v
        # actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Train
        self.NN.train()
        batch_times = []
        train_pred_times = []
        break_flag = False
        for i in range(self.constants['ppo']['optimization_epochs']):
            # Sync. at start of each epoch
            self._copy_shared_model_to_local()
            train_lstm_state = (
                    torch.zeros(1, 10, 128),
                    torch.zeros(1, 10, 128),
                )

            start_batch_time = time.time()

            batch_indices = np.arange(len(advantages))
            batch_indices = tensor(batch_indices, self.device).long()

            # Important Node: these are tensors but dont have a grad

            sampled_states = {}
            for k, v in states_.items():
                sampled_states[k] = v[batch_indices]
            unava_phase_indexs_sample = []
            for ind in batch_indices:
                unava_phase_indexs_sample.append(unava_phase_indexs[ind])
            sampled_actions = torch.LongTensor(actions)[:, batch_indices]
            # sampled_actions = actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices]
            sampled_advantages = advantages[batch_indices]

            start_pred_time = time.time()
            prediction, _, logits = self._get_prediction(sampled_states, train_lstm_state,
                                                 unava_phase_indexs_sample,
                                                 sampled_actions)
            end_pred_time = time.time()
            train_pred_times.append(end_pred_time - start_pred_time)

            # Calc. Loss
            logratio = prediction['log_pi_a'] - sampled_log_probs_old
            ratio = logratio.exp()

            if self.constants['ppo'].get("target_kl") is not None:
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()

                if approx_kl > self.constants['ppo']['target_kl']:
                    break

            obj = ratio * sampled_advantages
            obj_clipped = ratio.clamp(1.0 - self.constants['ppo']['ppo_ratio_clip'],
                                      1.0 + self.constants['ppo']['ppo_ratio_clip']) * sampled_advantages

            # supervise loss
            # sdk_action = sdk(sampled_states)
            # sdk_label, mask_sup = get_sdk_label(sampled_states,
            #                                     self.constants['environment'].get('single_min_phase_duration', 14) -
            #                                     self.constants['environment'].get('yellow_duration', 3))
            # supervise_loss = (torch.nn.CrossEntropyLoss(reduce=False)(logits.reshape(-1,4,3).permute(0,2,1), sdk_label.long()) * mask_sup).mean()
            # policy loss and value loss are scalars
            supervise_loss = 0
            policy_loss = -torch.min(obj, obj_clipped).mean()
            entropy_loss = -self.constants['ppo']['entropy_weight'] * prediction['ent'].mean()
            value_loss = self.constants['ppo']['value_loss_coef'] *  torch.nn.functional.mse_loss(sampled_returns, prediction['v'])
            self.opt.zero_grad()
            alpha = self.alpha
            (policy_loss + entropy_loss + value_loss).backward()
            # (policy_loss + entropy_loss + value_loss+ alpha * supervise_loss).backward()
            if self.constants['ppo']['clip_grads']:
                te = nn.utils.clip_grad_norm_(self.NN.parameters(),
                                              self.constants['ppo']['gradient_clip'])
            # for param in self.NN.parameters():
            #     print(param)
            #     if torch.isnan(param).any():
            #         print(1)
            ensure_shared_grads(self.NN, self.shared_NN)
            self.opt.step()
            end_batch_time = time.time()
            batch_times.append(end_batch_time - start_batch_time)
        self.NN.eval()
        print(policy_loss, entropy_loss, value_loss, supervise_loss)
        return policy_loss, entropy_loss, value_loss, supervise_loss, sampled_advantages.mean(), all_reward
