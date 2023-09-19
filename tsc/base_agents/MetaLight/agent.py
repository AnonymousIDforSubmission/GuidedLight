# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/10/12 16:41
 @Author  : 姜浩源
 @FileName: agent.py
 @Software: PyCharm
"""
import copy

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

import config
import time


def get_session(num_cpu):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1/10.
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)


class Agent():
    def __init__(self, config):
        t1 = time.time()
        self.config = config

        self._is_train = True
        self._alpha = self.config['ALPHA']
        self._min_alpha = self.config['MIN_ALPHA']
        self._alpha_decay_rate = self.config['ALPHA_DECAY_RATE']
        self._alpha_decay_step = self.config['ALPHA_DECAY_STEP']
        self._K = 1
        self._norm = self.config['NORM']#'None' #'batch_norm'
        self._batch_size = 20
        self._num_updates = self.config['NUM_UPDATES']
        self._avoid_second_derivative = False

        self._loss_fn = self._get_loss_fn('MSE')

        if self.config['ACTIVATION_FN'] == 'relu':
            self._activation_fn = tf.nn.relu
        elif self.config['ACTIVATION_FN'] == 'leaky_relu':
            self._activation_fn = tf.nn.leaky_relu
        else:
            raise(ValueError)

        ## dimension of input and output

        self.num_actions = 8

        self.num_phases = 8
        self.num_lanes = 2

        self.dim_input = 24
        # for feature_name in self.config["state_key"]:
        #     if "phase" in feature_name :
        #         self.dim_input += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0]*self.num_lanes*4
        #     else:
        #         self.dim_input += self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_"+feature_name.upper()][0]*self.num_lanes
        #

        self._weights = self.construct_weights(self.dim_input, self.num_actions)
        self._build_placeholder()
        self._build_graph(self.dim_input, self.num_actions, norm=self._norm)
        self._assign_op = [self._weights[key].assign(self._weights_inp[key]) for key in self._weights.keys()]
        self._meta_grads = dict(zip(self._weights.keys(), tf.gradients(self._meta_loss, list(self._weights.values()))))

        self._sess = get_session(1)
        self._sess.run(tf.global_variables_initializer())
        print("build policy time:", time.time() - t1)

    def _build_graph(self, dim_input, dim_output, norm):
        def model_summary():
            model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        learning_x, learning_y, meta_x, meta_y = [self._learning_x, self._learning_y,
                                  self._meta_x, self._meta_y]
        learning_loss_list = []
        meta_loss_list = []

        weights = self._weights
        learning_output = self.construct_forward(learning_x, weights,
                                                   reuse=False, norm=norm,
                                                   is_train=self._is_train)

        # Meta train loss: Calculate gradient
        learning_loss = self._loss_fn(learning_y, learning_output)
        learning_loss = tf.reduce_mean(learning_loss)
        learning_loss_list.append(learning_loss)
        grads = dict(zip(weights.keys(),
                         tf.gradients(learning_loss, list(weights.values()))))
        # learning rate
        self.learning_rate_op = tf.maximum(self._min_alpha,
                                           tf.train.exponential_decay(
                                               self._alpha,
                                               self.alpha_step,
                                               self._alpha_decay_step,
                                               self._alpha_decay_rate,
                                               staircase=True
                                           ))
        self.learning_train_op = tf.train.AdamOptimizer(self.learning_rate_op).minimize(learning_loss)
        if self.config['GRADIENT_CLIP']:
            for key in grads.keys():
                grads[key] = tf.clip_by_value(grads[key], -1 * self.config['CLIP_SIZE'], self.config['CLIP_SIZE'])

        self._learning_grads = grads
        new_weights = dict(zip(weights.keys(), [weights[key] - self.learning_rate_op * grads[key]
                                for key in weights.keys()]))

        if self._avoid_second_derivative:
            new_weights = tf.stop_gradients(new_weights)
        meta_output = self.construct_forward(meta_x, new_weights,
                                                 reuse=True, norm=norm,
                                                 is_train=self._is_train)
        # Meta val loss: Calculate loss (meta step)
        meta_loss = self._loss_fn(meta_y, meta_output)
        meta_loss = tf.reduce_mean(meta_loss)
        meta_loss_list.append(meta_loss)
        # If perform multiple updates

        for _ in range(self._num_updates - 1):
            learning_output = self.construct_forward(learning_x, new_weights,
                                                       reuse=True, norm=norm,
                                                       is_train=self._is_train)
            learning_loss = self._loss_fn(learning_y, learning_output)
            learning_loss = tf.reduce_mean(learning_loss)
            learning_loss_list.append(learning_loss)
            grads = dict(zip(new_weights.keys(),
                             tf.gradients(learning_loss, list(new_weights.values()))))
            new_weights = dict(zip(new_weights.keys(),
                                   [new_weights[key] - self.learning_rate_op * grads[key]
                                    for key in new_weights.keys()]))
            if self._avoid_second_derivative:
                new_weights = tf.stop_gradients(new_weights)
            meta_output = self.construct_forward(meta_x, new_weights,
                                                     reuse=True, norm=norm,
                                                     is_train=self._is_train)
            meta_loss = self._loss_fn(meta_y, meta_output)
            meta_loss = tf.reduce_mean(meta_loss)
            meta_loss_list.append(meta_loss)

        self._new_weights = new_weights

        # output
        self._learning_output = learning_output
        self._meta_output = meta_output

        # Loss
        learning_loss = tf.reduce_mean(learning_loss_list[-1])
        meta_loss = tf.reduce_mean(meta_loss_list[-1])

        self._learning_loss = learning_loss
        self._meta_loss = meta_loss
        # model_summary()

    def _get_loss_fn(self, loss_type):
        if loss_type == 'MSE':
            loss_fn = tf.losses.mean_squared_error
        else:
            ValueError("Can't recognize the loss type {}".format(loss_type))
        return loss_fn

    def learning_predict(self, learning_x):
        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._learning_x: learning_x
                }
                return self._sess.run(self._learning_output, feed_dict=feed_dict)

    def meta_predict(self, meta_x):
        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._meta_x: meta_x
                }
                return self._sess.run(self._meta_output, feed_dict=feed_dict)

    def _build_placeholder(self):
        self.alpha_step = tf.placeholder('int64', None, name='alpha_step')
        self._learning_x = tf.placeholder(tf.float32, shape=(None, self.dim_input))
        self._learning_y = tf.placeholder(tf.float32, shape=(None, self.num_actions))
        self._meta_x = tf.placeholder(tf.float32, shape=(None, self.dim_input))
        self._meta_y = tf.placeholder(tf.float32, shape=(None, self.num_actions))
        self._weights_inp = {}
        for key in self._weights.keys():
            self._weights_inp[key] = tf.placeholder(tf.float32, shape=self._weights[key].shape)

    def choose_action(self, state, test=False):
        ''' choose the best action for current state '''
        state = copy.deepcopy(state)
        unava = state[0]['unava']
        unava_n = []
        for i in unava:
            tmp = []
            for j in range(8):
                if j not in i:
                    tmp.append(1)
                else:
                    tmp.append(0)
            unava_n.append(tmp)
        # unava = np.array(unava_n)
        all_tls = state[0]['tls']
        # del state[0]['unava']
        # del state[0]['all_tls']
        inputs = [[] for _ in range(len(state[0])-2)]
        # all_start_lane = self.config["LANE_PHASE_INFO"]["start_lane"]
        for i, k in enumerate(all_tls):
            s = state[0][k]
            ss, mask = s
            stop_car_nums = []
            current_phases = []
            for j in ss:
                stop_car_nums.append(j.get('stop_car_num',0))
                current_phases.append(j.get('current_phase',0))
            inputs[i].extend(stop_car_nums + current_phases + unava_n[i])
        # for i in range(len(state)):
        #     s = state[i]
        #     inputs[i].extend(s['lane_num_vehicle'] + s["cur_phase"])
        inputs = np.reshape(np.array(inputs), (len(inputs), -1))
        q_values = self.learning_predict(inputs)

        if not test:
            if random.random() <= self.config["EPSILON"]:  # continue explore new Random Action
                action = np.array(
                    [random.choice(
                        [i for i in range(q_values.shape[1]) if i not in unava[j]]
                    ) for j in range(q_values.shape[0])]
                )
            else:  # exploitation
                action = np.argmax(q_values, axis=1)  # q_values shape: (2, 1, 8)
        else:
            action = np.argmax(q_values, axis=1)

        return action * 2

    def decay_epsilon(self, batch_id):
        decayed_epsilon = self.config["EPSILON"] * pow(self.config["EPSILON_DECAY"], batch_id)
        self.config["EPSILON"] = max(decayed_epsilon, self.config["MIN_EPSILON"])

    def fit(self, episodes, params, target_params):
        self.load_params(params)
        input_x = episodes.get_x()
        inputs = [[] for _ in range(len(input_x))]
        for i, (x, unava) in enumerate(input_x):
            unava_n = []
            for ii in range(8):
                if ii in unava:
                    unava_n.append(0)
                else:
                    unava_n.append(1)
            stop_car_nums = []
            current_phases = []
            for j in x[0]:
                stop_car_nums.append(j.get('stop_car_num', 0))
                current_phases.append(j.get('current_phase', 0))
            inputs[i].extend(stop_car_nums + current_phases + unava_n)
        q_values = self.learning_predict(inputs)

        self.load_params(target_params)
        input_next_x = episodes.get_next_x()
        inputs_next = [[] for _ in range(len(input_next_x))]
        for i, (x, unava) in enumerate(input_next_x):
            unava_n = []
            for ii in range(8):
                if ii in unava:
                    unava_n.append(0)
                else:
                    unava_n.append(1)
            stop_car_nums = []
            current_phases = []
            for j in x[0]:
                stop_car_nums.append(j.get('stop_car_num', 0))
                current_phases.append(j.get('current_phase', 0))
            inputs_next[i].extend(stop_car_nums + current_phases + unava_n)
        target_q_values = self.learning_predict(inputs_next)

        for i in range(len(episodes.total_samples)):
            sample = episodes.total_samples[i]
            action = sample[1] // 2
            reward = sample[3]
            q_values[i][action] = reward['queue_len'] * 1000 + self.config['GAMMA'] * np.max(target_q_values[i])

        episodes.prepare_y(q_values)

    def update_params(self, episodes, params, lr_step, slice_index):
        learning_x = episodes.get_x()[slice_index]
        learning_y = episodes.get_y()[slice_index]
        inputs = [[] for _ in range(len(learning_x))]
        for i, (x, unava) in enumerate(learning_x):
            unava_n = []
            for ii in range(8):
                if ii in unava:
                    unava_n.append(0)
                else:
                    unava_n.append(1)
            stop_car_nums = []
            current_phases = []
            for j in x[0]:
                stop_car_nums.append(j.get('stop_car_num', 0))
                current_phases.append(j.get('current_phase', 0))
            inputs[i].extend(stop_car_nums + current_phases + unava_n)
        print("Task | Traffic:", self.config['sumocfg_files'])
        t1 = time.time()

        if self.config['OPTIMIZER'] == 'sgd':
            for i in range(self.config['NUM_GRADIENT_STEP']):
                self.load_params(params)
                with self._sess.as_default():
                    with self._sess.graph.as_default():
                        feed_dict = {
                            self._learning_x: inputs,
                            self._learning_y: learning_y,
                            self.alpha_step: lr_step
                        }
                        params, learning_loss, lr = self._sess.run([self._new_weights, self._learning_loss, self.learning_rate_op], feed_dict=feed_dict)
                        print("step: %d, epoch: %3d, loss: %f, learning_rate: %f, epsilon: %f" % (
                            lr_step, i, learning_loss, lr, self.config["EPSILON"]))
        elif self.config['OPTIMIZER'] == 'adam':
            _weights_list = list(self._weights.values())

            for i in range(self.config['NUM_GRADIENT_STEP']):
                with self._sess.as_default():
                    with self._sess.graph.as_default():
                        feed_dict = {
                            self._learning_x: inputs,
                            self._learning_y: learning_y,
                            self.alpha_step: lr_step
                        }
                        _, weights_list, learning_loss, lr = self._sess.run([self.learning_train_op, _weights_list, self._learning_loss, self.learning_rate_op], feed_dict=feed_dict)
                        print("step: %d, epoch: %3d, loss: %f, learning_rate: %f, epsilon: %f" % (
                            lr_step, i, learning_loss, lr, self.config["EPSILON"]))
            params = dict(zip(self._weights.keys(), weights_list))
        else:
            raise(NotImplementedError)
        t2 = time.time()
        return params

    def load_params(self, params):
        with self._sess.as_default():
           with self._sess.graph.as_default():
               feed_dict = {self._weights_inp[key]: params[key] for key in self._weights.keys()}
               self._sess.run(self._assign_op, feed_dict=feed_dict)

    def save_params(self):
        with self._sess.as_default():
            with self._sess.graph.as_default():
                return self._sess.run(self._weights)

    def cal_grads(self, learning_episodes, meta_episodes, slice_index, params):
        self.load_params(params)
        t1 = time.time()

        if not second_index:
            second_index = slice_index

        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._learning_x: learning_episodes.get_x()[slice_index],
                    self._learning_y: learning_episodes.get_y()[slice_index],
                    self._meta_x: meta_episodes.get_x()[second_index],
                    self._meta_y: meta_episodes.get_y()[second_index],
                    self.alpha_step: 0, # TODO hard code
                }
                res = self._sess.run(self._meta_grads, feed_dict=feed_dict)
        t2 = time.time()
        return res

    def second_cal_grads(self, episodes, slice_index, new_slice_index, params):
        self.load_params(params)
        t1 = time.time()
        _learning_x = episodes.get_x()[slice_index]
        _learning_x_inputs = [[] for _ in range(len(_learning_x))]
        for i, (x, unava) in enumerate(_learning_x):
            unava_n = []
            for ii in range(8):
                if ii in unava:
                    unava_n.append(0)
                else:
                    unava_n.append(1)
            stop_car_nums = []
            current_phases = []
            for j in x[0]:
                stop_car_nums.append(j.get('stop_car_num', 0))
                current_phases.append(j.get('current_phase', 0))
            _learning_x_inputs[i].extend(stop_car_nums + current_phases + unava_n)

        _meta_x = episodes.get_x()[new_slice_index]
        _meta_x_inputs = [[] for _ in range(len(_meta_x))]
        for i, (x, unava) in enumerate(_meta_x):
            unava_n = []
            for ii in range(8):
                if ii in unava:
                    unava_n.append(0)
                else:
                    unava_n.append(1)
            stop_car_nums = []
            current_phases = []
            for j in x[0]:
                stop_car_nums.append(j.get('stop_car_num', 0))
                current_phases.append(j.get('current_phase', 0))
            _meta_x_inputs[i].extend(stop_car_nums + current_phases + unava_n)


        with self._sess.as_default():
            with self._sess.graph.as_default():
                feed_dict = {
                    self._learning_x: _learning_x_inputs,
                    self._learning_y: episodes.get_y()[slice_index],
                    self._meta_x: _meta_x_inputs,
                    self._meta_y: episodes.get_y()[new_slice_index],
                    self.alpha_step: 0,  # TODO hard code
                }
                res = self._sess.run(self._meta_grads, feed_dict=feed_dict)
        t2 = time.time()
        return res