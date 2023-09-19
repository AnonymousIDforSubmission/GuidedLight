# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:41
 @Author  : 姜浩源
 @FileName: frapplus_agent.py
 @Software: PyCharm
"""
import tensorflow as tf
import numpy as np

from agent import Agent


def contruct_layer(inp, activation_fn, reuse, norm, is_train, scope):
    if norm == 'batch_norm':
        out = tf.contrib.layers.batch_norm(inp, activation_fn=activation_fn,
                                           reuse=reuse, is_training=is_train,
                                           scope=scope)
    elif norm == None:
        out = activation_fn(inp)
    else:
        ValueError('Can\'t recognize {}'.format(norm))
    return out


def relation(phase_list):
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
    else:
        relations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape((1, 4, 3))
    constant = relations
    return constant

class FRAPPlusAgent(Agent):
    """
        FRAP++ makes a few improvements to FRAP (https://github.com/gjzheng93/frap-pub) and is also a Tensorflow version
    """
    def construct_weights(self, dim_input, dim_output):
        weights = {}

        weights['embed_w1'] = tf.Variable(tf.glorot_uniform_initializer()([1, 4]), name='embed_w1')
        weights['embed_b1'] = tf.Variable(tf.zeros([4]), name='embed_b1')

        # for phase, one-hot
        weights['embed_w2'] = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05)([2, 4]), name='embed_w2')

        # lane embeding
        weights['lane_embed_w3'] = tf.Variable(tf.glorot_uniform_initializer()([8, 16]), name='lane_embed_w3')
        weights['lane_embed_b3'] = tf.Variable(tf.zeros([16]), name='lane_embed_b3')

        # relation embeding, one-hot
        weights['relation_embed_w4'] = tf.Variable(tf.random_uniform_initializer(minval=-0.05, maxval=0.05)([2, 4]), name='relation_embed_w4')

        weights['feature_conv_w1'] = tf.Variable(tf.glorot_uniform_initializer()([1, 1, 32, self.config["D_DENSE"]]), name='feature_conv_w1')
        weights['feature_conv_b1'] = tf.Variable(tf.zeros([self.config['D_DENSE']]), name='feature_conv_b1')

        weights['phase_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, 4, self.config["D_DENSE"]]), name='phase_conv_w1')
        weights['phase_conv_b1'] = tf.Variable(tf.zeros([self.config['D_DENSE']]), name='phase_conv_b1')

        weights['combine_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, self.config["D_DENSE"], self.config["D_DENSE"]]), name='combine_conv_w1')
        weights['combine_conv_b1'] = tf.Variable(tf.zeros([self.config['D_DENSE']]), name='combine_conv_b1')

        weights['final_conv_w1'] = tf.Variable(
            tf.glorot_uniform_initializer()([1, 1, self.config["D_DENSE"], 1]), name='final_conv_w1')
        weights['final_conv_b1'] = tf.Variable(tf.zeros([1]), name='final_conv_b1')

        return weights

    def construct_forward(self, inp, weights, reuse, norm, is_train, prefix='fc'):

        dim = int((inp.shape[1].value - 8) / 2)
        num_veh = inp[:, :dim]
        batch_size = num_veh.shape[0]
        num_veh = tf.reshape(num_veh, [-1, 1])

        phase = inp[:, dim:(inp.shape[1].value - 8)]
        phase = tf.cast(phase, tf.int32)
        phase = tf.one_hot(phase, 2)
        phase = tf.reshape(phase, [-1, 2])

        unava = inp[:, -8:]

        embed_num_veh = contruct_layer(tf.matmul(num_veh, weights['embed_w1']) + weights['embed_b1'],
                                 activation_fn=tf.nn.sigmoid, reuse=reuse, is_train=is_train,
                                 norm=norm, scope='num_veh_embed.' + prefix
                                 )
        embed_num_veh = tf.reshape(embed_num_veh, [-1, dim, 4])

        embed_phase = contruct_layer(tf.matmul(phase, weights['embed_w2']),
                                 activation_fn=tf.nn.sigmoid, reuse=reuse, is_train=is_train,
                                 norm=norm, scope='phase_embed.' + prefix
                                 )
        embed_phase = tf.reshape(embed_phase, [-1, dim, 4])

        dic_lane = []
        for i in range(8):
            dic_lane.append(tf.concat([embed_num_veh[:, i, :], embed_phase[:, i, :]], axis=-1))


        list_phase_pressure = []
        phase_startLane_mapping = [[3,7],[6,7],[2,3],[2,6],[1,5],[4,5],[0,1],[0,4]]
        for a, b in phase_startLane_mapping:
            t1 = tf.Variable(tf.zeros(1))

            for lane in [a, b]:
                t1 += contruct_layer(
                   tf.matmul(dic_lane[lane], weights['lane_embed_w3']) + weights['lane_embed_b3'],
                   activation_fn=self._activation_fn, reuse=reuse, is_train=is_train,
                   norm=norm, scope='lane_embed.' + prefix
                   )
            t1 /= 2
            list_phase_pressure.append(t1)


        constant = relation([
            'WT_ET',
            'EL_ET',
            'WL_WT',
            'WL_EL',
            'NT_ST',
            'SL_ST',
            'NT_NL',
            'NL_SL'
        ])

        constant = tf.one_hot(constant, 2)
        s1, s2 = constant.shape[1:3]
        constant = tf.reshape(constant, (-1, 2))
        relation_embedding = tf.matmul(constant, weights['relation_embed_w4'])
        relation_embedding = tf.reshape(relation_embedding, (-1, s1, s2, 4))

        list_phase_pressure_recomb = []
        num_phase = len(list_phase_pressure)

        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    list_phase_pressure_recomb.append(
                        tf.concat([list_phase_pressure[i], list_phase_pressure[j]], axis=-1,
                                    name="concat_compete_phase_%d_%d" % (i, j)))

        list_phase_pressure_recomb = tf.concat(list_phase_pressure_recomb, axis=-1 , name="concat_all")
        feature_map = tf.reshape(list_phase_pressure_recomb, (-1, num_phase, num_phase-1, 32))

        lane_conv = tf.nn.conv2d(feature_map, weights['feature_conv_w1'], [1, 1, 1, 1], 'VALID', name='feature_conv') + weights['feature_conv_b1']
        lane_conv = tf.nn.leaky_relu(lane_conv, name='feature_activation')

        # relation conv layer
        relation_conv = tf.nn.conv2d(relation_embedding, weights['phase_conv_w1'], [1, 1, 1, 1], 'VALID',
                                 name='phase_conv') + weights['phase_conv_b1']
        relation_conv = tf.nn.leaky_relu(relation_conv, name='phase_activation')
        combine_feature = tf.multiply(lane_conv, relation_conv, name="combine_feature")

        # second conv layer
        hidden_layer = tf.nn.conv2d(combine_feature, weights['combine_conv_w1'], [1, 1, 1, 1], 'VALID', name='combine_conv') + \
                    weights['combine_conv_b1']
        hidden_layer = tf.nn.leaky_relu(hidden_layer, name='combine_activation')

        before_merge = tf.nn.conv2d(hidden_layer, weights['final_conv_w1'], [1, 1, 1, 1], 'VALID',
                                    name='final_conv') + \
                       weights['final_conv_b1']

        _shape = (-1, self.num_actions, self.num_actions-1)
        before_merge = tf.reshape(before_merge, _shape)
        out = tf.reduce_sum(before_merge, axis=2)

        a = out * unava - 1e8 * (1-unava)


        return a

    def init_params(self):
        return self.save_params()
