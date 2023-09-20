#!/usr/bin/env python3
# encoding: utf-8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_

import numpy as np


def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def conv2d_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    pad_type='zero',
    activation=None,
):
    block = []
    assert pad_type in ['zero', 'reflect', 'replication'], "invalid padding type: {}".format(pad_type)
    if pad_type == 'zero':
        pass
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
        padding = 0
    elif pad_type == 'replication':
        block.append(nn.ReplicationPad2d(padding))
        padding = 0
    block.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups)
    )
    xavier_normal_(block[-1].weight)
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def fc_block(
        in_channels,
        out_channels,
        activation=None,
        use_dropout=False,
        norm_type=None,
        dropout_probability=0.5
):
    block = [nn.Linear(in_channels, out_channels)]
    xavier_normal_(block[-1].weight)
    if norm_type is not None and norm_type != 'none':
        if norm_type == 'LN':
            block.append(nn.LayerNorm(out_channels))
        else:
            raise NotImplementedError
    if isinstance(activation, torch.nn.Module):
        block.append(activation)
    elif activation is None:
        pass
    else:
        raise NotImplementedError
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class ModelBody(nn.Module):
    def __init__(self, input_size, fc_layer_size, state_keys, device='cpu'):
        self.state_keys = state_keys
        super(ModelBody, self).__init__()
        self.name = 'model_body'

        # mlp
        self.fc_flow = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        if "queue_length" in state_keys:
                self.fc_queue_length = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_spillback = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_road_capacity = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        # # current phase
        # self.current_phase_act = nn.Sigmoid()
        # self.current_phase_embedding = nn.Embedding(2, 4)
        # mask
        self.mask_act = nn.Sigmoid()
        self.mask_embedding = nn.Embedding(2, 4)
        # dirct liner
        self.dirct_fc = fc_block(30, 32, activation=nn.Sigmoid())
        # relation_embedding
        self.relation_embedding = nn.Embedding(2, 16)
        self.PHASE_LIST = [
            'WT_ET',
            'EL_ET',
            'WL_WT',
            'WL_EL',
            'NT_ST',
            'SL_ST',
            'NT_NL',
            'NL_SL'
        ]

        # self.constant = self.relation(PHASE_LIST, device)
        # self.constant =  torch.LongTensor([[[0, 0, 0, 1, 1, 0, 0],
        #                                     [0, 0, 0, 0, 0, 1, 1],
        #                                     [0, 0, 0, 1, 1, 0, 0],
        #                                     [0, 0, 0, 0, 0, 1, 1],
        #                                     [1, 0, 1, 0, 0, 0, 0],
        #                                     [1, 0, 1, 0, 0, 0, 0],
        #                                     [0, 1, 0, 1, 0, 0, 0],
        #                                     [0, 1, 0, 1, 0, 0, 0]]], device=device)
        # cnn, as well as fc
        if 'queue_length' in self.state_keys:
            self.drict_cnn = conv2d_block(48, 32, 1, activation=nn.ReLU())
        else:
            self.drict_cnn = conv2d_block(40, 32, 1, activation=nn.ReLU())
        self.relation_cnn = conv2d_block(16, 32, 1, activation=nn.ReLU())
        self.cnn = conv2d_block(32, 16, 1, activation=nn.ReLU())
        self.lstm = nn.LSTM(80, 128, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.pre = conv2d_block(16, 16, 1)

    def relation(self, phase_list, device='cpu'):
        relations = []
        num_phase = len(phase_list)
        if num_phase == 8:
            for p1 in phase_list:
                zeros = [0, 0, 0, 0, 0, 0, 0]
                count = 0
                for p2 in phase_list:
                    if p1 == p2:
                        continue
                    m1 = p1.split("_")
                    m2 = p2.split("_")
                    if len(list(set(m1 + m2))) == 3:
                        zeros[count] = 1
                    count += 1
                relations.append(zeros)
            relations = np.array(relations).reshape((1, 8, 7))
        elif num_phase == 4:
            for p1 in phase_list:
                zeros = [0, 0, 0]
                count = 0
                for p2 in phase_list:
                    if p1 == p2:
                        continue
                    m1 = p1.split("_")
                    m2 = p2.split("_")
                    if len(list(set(m1 + m2))) == 3:
                        zeros[count] = 1
                    count += 1
                relations.append(zeros)
            relations = np.array(relations).reshape((1, 4, 3))
        else:
            relations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape((1, 4, 3))
        constant = torch.LongTensor(relations, device=device)
        return constant

    def forward(self, input, lstm_state, unava_phase_index):
        # if len(lstm_state) == 3:
        #     lstm_state = (lstm_state[0].unsqueeze(0), lstm_state[1].unsqueeze(0))
        all_key_state = []
        bs = (list(input.values())[0].shape)[0]
        for i in self.state_keys:
            if i in ['current_phase', 'mask', 'green_utilization', 'green_balance',
                     'duration', 'phase_index', 'masked_phase']:
                continue
            all_key_state.append(eval("self.fc_{}".format(i))(input[i].reshape(-1, 1)))
        if "current_phase" in self.state_keys:
            all_key_state.append(input['current_phase'].permute(0, 2, 1).reshape(-1, 4))
        all_key_state.append(self.mask_act(
            self.mask_embedding(input['mask'])).reshape(-1, 4))
        direct_all = torch.cat(all_key_state, dim=1).reshape(bs, 8, -1)
        mix_direct = torch.cat(
            [
            torch.add(direct_all[:, 3, :], direct_all[:, 7, :]).unsqueeze(1),
            torch.add(direct_all[:, 6, :], direct_all[:, 7, :]).unsqueeze(1),
            torch.add(direct_all[:, 2, :], direct_all[:, 3, :]).unsqueeze(1),
            torch.add(direct_all[:, 2, :], direct_all[:, 6, :]).unsqueeze(1),
            torch.add(direct_all[:, 1, :], direct_all[:, 5, :]).unsqueeze(1),
            torch.add(direct_all[:, 4, :], direct_all[:, 5, :]).unsqueeze(1),
            torch.add(direct_all[:, 0, :], direct_all[:, 1, :]).unsqueeze(1),
            torch.add(direct_all[:, 0, :], direct_all[:, 4, :]).unsqueeze(1)
            ]
            , dim=1)  #  A: wt-et B: el-et C: wl-wt D: el-wl E: nt-st F: sl-st G: nt-nl H: nl-sl

        mix_direct_fitter = []
        for i in range(bs):
            mix_direct_fitter.append(mix_direct[i, input['phase_index'][i], :])
        mix_direct_fitter = torch.stack(mix_direct_fitter)
        list_phase_pressure_recomb = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    list_phase_pressure_recomb.append(
                        torch.cat([mix_direct_fitter[:, i, :], mix_direct_fitter[:, j, :]], dim=-1).unsqueeze(1))
        list_phase_pressure_recomb = torch.cat(
            list_phase_pressure_recomb, dim=1).reshape(bs, 4, 3, -1).permute(0, 3, 1, 2)
        constant = []
        for i in range(bs):
            constant.append(self.relation(np.array(self.PHASE_LIST)[input['phase_index'][i]], 'cpu'))
        constant = torch.cat(constant)
        relation_embedding = self.relation_embedding(constant).permute(0, 3, 1, 2)

        direct_cnn = self.drict_cnn(list_phase_pressure_recomb)
        relation_conv = self.relation_cnn(relation_embedding)
        combine_feature = direct_cnn * relation_conv
        hidden_layer = self.cnn(combine_feature)

        hidden = self.pre(hidden_layer).sum(-1)
        # output = self.output(hidden_layer).sum(-1).squeeze(1)
        #
        # if unava_phase_index:
        #     for i in range(bs):
         #         output[i, unava_phase_index[i]] = 0
        phase_index = torch.argsort(input['phase_index'], dim=1)
        green_utilization = torch.gather(input['green_utilization'], dim=1, index=phase_index)
        green_balance = torch.gather(input['green_balance'], dim=1, index=phase_index)
        duration = torch.gather(input['duration'], dim=1, index=phase_index)
        mask_phase = torch.gather(input['masked_phase'], dim=1, index=phase_index)
        obs = torch.cat([hidden, green_utilization.unsqueeze(1),
                         green_balance.unsqueeze(1), duration.unsqueeze(1),
                         mask_phase.unsqueeze(1)], dim=1)
        new_hidden = []
        obs = obs.reshape((10, -1, 80))
        if len(lstm_state[0].shape) == 2:
            h, lstm_state = self.lstm(obs.unsqueeze(1),
                                      (lstm_state[0].unsqueeze(0), lstm_state[1].unsqueeze(0)))
        else:
            h, lstm_state = self.lstm(obs, (lstm_state[0], lstm_state[1]))
        new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state


class ActorModel(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(64, action_size)
        )

    def forward(self, hidden_states, inputs):
        outputs = self.model(hidden_states)
        return outputs


class CriticModel(nn.Module):
    def __init__(self, hidden_size):
        super(CriticModel, self).__init__()
        self.name = 'critic'
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, hidden_states, inputs):
        outputs = self.model(hidden_states)
        return outputs


class NN_Model(nn.Module):
    def __init__(self, action_size, hidden_layer_size, state_keys, device='cpu'):
        super(NN_Model, self).__init__()
        self.state_keys = state_keys
        if action_size == 12:
            self.group = True
        else:
            self.group = False
        self.body_model = ModelBody(1, hidden_layer_size, self.state_keys, device=device).to(device)
        self.actor_model = ActorModel(hidden_layer_size * 2, action_size).to(device)
        self.critic_model = CriticModel(hidden_layer_size * 2).to(device)

    # def initial_state(self, batch_size=1):
    #     return torch.zeros(2, batch_size, 512)

    def forward(self, inputs, lstm_state, unava_phase_index, actions=None, eval=False):
        hidden_states, lstm_state = self.body_model(inputs, lstm_state, unava_phase_index)
        v = self.critic_model(hidden_states, inputs)
        logits = self.actor_model(hidden_states, inputs)
        # phase_index = torch.argsort(inputs['phase_index'], dim=1)
        # phase_index = torch.argsort(phase_index, dim=1)  # do twice
        # phase_index *= 2
        # phase_index = phase_index.repeat_interleave(2, dim=-1) + torch.tensor([0, 1]).repeat(phase_index.shape)
        # logits = torch.gather(logits, dim=1, index=phase_index)
        if self.group:
            if unava_phase_index:
                for i in range(hidden_states.shape[0]):
                    if len(unava_phase_index[i]) > 0:
                        una = np.array(unava_phase_index[i]) * 3
                        una = una.repeat(2) + np.tile(np.array([0, 1]), len(unava_phase_index[i]))
                        logits[i, una] = -1e8
            dist1 = torch.distributions.Categorical(logits=logits[:,:3])
            dist2 = torch.distributions.Categorical(logits=logits[:,3:6])
            dist3 = torch.distributions.Categorical(logits=logits[:,6:9])
            dist4 = torch.distributions.Categorical(logits=logits[:,9:])
            if actions is None:
                if not eval:
                    actions1 = dist1.sample()
                    actions2 = dist2.sample()
                    actions3 = dist3.sample()
                    actions4 = dist4.sample()
                else:
                    actions1 = torch.argmax(logits[:,:3], axis=1).detach()
                    actions2 = torch.argmax(logits[:,3:6], axis=1).detach()
                    actions3 = torch.argmax(logits[:,6:9], axis=1).detach()
                    actions4 = torch.argmax(logits[:, 9:], axis=1).detach()
                actions = [actions1, actions2, actions3, actions4]
            log_prob1 = dist1.log_prob(actions[0]).unsqueeze(-1)
            log_prob2 = dist2.log_prob(actions[1]).unsqueeze(-1)
            log_prob3 = dist3.log_prob(actions[2]).unsqueeze(-1)
            log_prob4 = dist4.log_prob(actions[3]).unsqueeze(-1)
            log_prob = log_prob1 + log_prob2 + log_prob3 + log_prob4

            entropy = dist1.entropy().unsqueeze(-1) + dist2.entropy().unsqueeze(-1) + \
                      dist3.entropy().unsqueeze(-1) + dist4.entropy().unsqueeze(-1)

        else:
            if unava_phase_index:
                for i in range(hidden_states.shape[0]):
                    if len(unava_phase_index[i]) > 0:
                        una = np.array(unava_phase_index[i]) * 2
                        una = una.repeat(2) + np.tile(np.array([0, 1]), len(unava_phase_index[i]))
                        logits[i, una] = -1e8
            dist = torch.distributions.Categorical(logits=logits)
            if actions is None:
                if not eval:
                    actions = dist.sample()
                else:
                    actions = torch.argmax(logits, axis=1)
            log_prob = dist.log_prob(actions).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
        return {'a': actions,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}, lstm_state, logits


if __name__ == '__main__':
    from sumo_files.env.sim_env import TSCSimulator
    from utils import batch

    config = {
        "name": "test",
        "agent": "",
        "sumocfg_file": "sumo_files/scenarios/nanshan/osm.sumocfg",
        # "sumocfg_file": "sumo/scenarios/large_grid2/exp_0.sumocfg",
        # "sumocfg_file": "sumo/scenarios/sumo_wj3/rl_wj.sumocfg",
        # "sumocfg_file": "sumo/scenarios/real_net/most_0.sumocfg",
        # "sumocfg_file": "sumo/scenarios/sumo_fenglin_base_road/base.sumocfg",
        "action_type": "select_phase",
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'state_key': ['current_phase', 'car_num', 'queue_length', "occupancy", 'flow', 'stop_car_num']
    }
    env = TSCSimulator(config, 123)
    next_obs = env.reset()
    all_tl = env.all_tls
    model = NN_Model(12, 4, state_keys=config['state_key'])
    batch_ph = env.tl_phase_index
    for i in range(100):
        batch_next_obs = batch(next_obs, config['state_key'], all_tl)
        action = model(batch_next_obs)
        tl_action_select = {}
        for tl_index in range(len(all_tl)):
            tl_action_select[all_tl[tl_index]] = (env._crosses[all_tl[tl_index]].green_phases)[action['a'][tl_index]]
        next_obs, reward, done = env.step(tl_action_select)
    env.terminate()
