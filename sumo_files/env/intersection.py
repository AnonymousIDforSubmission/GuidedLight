#!/usr/bin/env python3
# encoding: utf-8

import copy

import numpy as np
import re
from tsc.utils import get_vector_cos_sim
import random


class Intersection:
    def __init__(self, tl_id, msg, env,
                 state=['current_phase', 'car_num', 'queue_length', "occupancy", 'flow',
                        'stop_car_num'], not_default=True):
        self._id = tl_id
        self._env = env
        self.state = state
        self.nearset_inter = msg['adjacency_row']
        self.location = msg.get('location')
        self.yellow_time = env.config['yellow_duration']
        self.single_min_phase_duration = env.config.get('single_min_phase_duration', 14) - self.yellow_time
        self.single_max_phase_duration = env.config.get('single_max_phase_duration', 45)

        self.car_passtime = env.config.get('car_passtime', 2)  # or 3， 后期根据拥堵情况自适应
        self.spillback_threshold = env.config.get('spillback_threshold', 0.2)  # 考虑到车辆之间的间隔，阈值设置为0.6
        self.halting_threshold = env.config.get('halting_threshold', 0.3)  # 车速低于0.1的车辆数比总车辆数，阈值设置为0.9

        self._incoming_lanes = list(
            set(self._env.sim.trafficlight.getControlledLanes(self._id)))
        try:
            self._outgoing_lanes = list(
                set([l[0][1] for l in self._env.sim.trafficlight.getControlledLinks(self._id)]))
        except:
            self.tl_ava = False
            return
        # seq = ["r", "s", "l"]  right always can go though
        self.seq = ["l", "s"]

        self.lans_link = self._env.sim.trafficlight.getControlledLinks(self._id)  # from : to: [traffic light index, direction]
        self._incominglanes_links = {}
        try:
            for index, [(fr, to, _)] in enumerate(self.lans_link):
                if fr not in self._incominglanes_links:
                    self._incominglanes_links[fr] = {to: [index]}
                elif to not in self._incominglanes_links[fr]:
                    self._incominglanes_links[fr][to] = [index]
                else:
                    self._incominglanes_links[fr][to].append(index)
        except:
            self.tl_ava = False
            return
        for in_line in self._incominglanes_links:
            msgs = self._env.sim.lane.getLinks(in_line)
            for m in msgs:
                self._incominglanes_links[in_line][m[0]].append(m[-2])

        self.entering_approaches2phase = {}
        for i in self._incoming_lanes:
            approach = "_".join(i.split("_")[:-1])
            if approach not in self.entering_approaches2phase:
                self.entering_approaches2phase[approach] = {}
            phase_index, directions = list(zip(*list(self._incominglanes_links[i].values())))
            if 's' in directions:
                if 's' not in self.entering_approaches2phase[approach]:
                    self.entering_approaches2phase[approach]['s'] = []
                self.entering_approaches2phase[approach]['s'].append([i, phase_index])
            elif "l" in directions or "L" in directions or 't' in directions and len(
                    directions) == 1:
                if 'l' not in self.entering_approaches2phase[approach]:
                    self.entering_approaches2phase[approach]['l'] = []
                self.entering_approaches2phase[approach]['l'].append([i, phase_index])
            elif "r" in directions or "R" in directions:
                if 'r' not in self.entering_approaches2phase[approach]:
                    self.entering_approaches2phase[approach]['r'] = []
                self.entering_approaches2phase[approach]['r'].append([i, phase_index])
            else:
                print(directions)
                print("Not support")
        self.entering_sequence = list(self.entering_approaches2phase.keys())

        self.outgoing_sequence = list("_".join(i.split("_")[:-1]) for i in msg['leaving_lanes'])

        order_entering = []
        for i in self.entering_sequence:
            order_entering.append(msg['entering_lanes'].index(i + "_0"))

        order_outgoing = []
        for i in self.outgoing_sequence:
            order_outgoing.append(msg['leaving_lanes'].index(i + "_0"))

        self.tl_ava, self.entering_sequence_NWSE = self._coordinate_sequence(
            msg["entering_lanes_pos"], order_entering, True)

        tl_av, self.outgoing_sequence_NWSE = self._coordinate_sequence(
            msg["leaving_lanes_pos"], order_outgoing, False)
        # assert tl_av == self.tl_ava
        self.tl_ava = tl_av and self.tl_ava
        if self.tl_ava:
            self._phase_index()

            self._lane_vehicle_dict = {}
            self._previous_lane_vehicle_dict = {}

            self.unava_index = self.reset_tl_logic()
            signal_definition = self._env.sim.trafficlight.getAllProgramLogics(self._id)[-1]
            self._env.sim.trafficlight.setProgram(self._id, "custom")
            # self.green_phases = [i for i in range(0, 16, 2)]
            # self.yellow_phases = [i for i in range(1, 16, 2)]
            self.phases = []
            for i in signal_definition.phases:
                self.phases.append(i.state)

        self.last_green_utilization = None
        self.last_flow = None
        self.last_queue_length = None
        self.history_phases = []
        self.history_duration = []
        self.last_spillback = None

    def get_tl_ava(self):
        return self.tl_ava

    def _phase_index(self):
        """get_phase_index
        The phase index is : north(ls), west(ls), south(ls), east(ls)
        """
        self.phase_index = []
        for i in self.entering_sequence_NWSE:
            if i != -1:
                seq_i = self.entering_sequence[i]
                for s in self.seq:
                    _phase = []
                    if s in self.entering_approaches2phase[seq_i]:
                        for l in self.entering_approaches2phase[seq_i][s]:
                            _phase.extend(list((l[1])))
                        self.phase_index.append(_phase)
                    else:
                        self.phase_index.append([])
            else:
                for _ in self.seq:
                    self.phase_index.append([])

    def get_phase_index(self):
        return self.phase_index

    @staticmethod
    def _coordinate_sequence(list_coord_str, order, entering):
        """Get each approach direction.
        Result sequence is north, west, south, east."""
        dim = len(list_coord_str[0].split(" ")[0].split(","))
        if entering:
            list_coordinate = [re.split(r'[ ,]', lane_str)[-2 * dim:]
                               for lane_str in list_coord_str]
            list_coordinate = np.array(list_coordinate, dtype=float)[order]
            if dim == 3:
                list_coordinate = np.concatenate([list_coordinate[:, :2], list_coordinate[:, 3:5]],
                                                 axis=1)
            delta_x = list_coordinate[:, 2] - list_coordinate[:, 0]
            delta_y = list_coordinate[:, 3] - list_coordinate[:, 1]
        else:
            list_coordinate = [re.split(r'[ ,]', lane_str)[: 2 * dim]
                               for lane_str in list_coord_str]
            list_coordinate = np.array(list_coordinate, dtype=float)[order]
            if dim == 3:
                list_coordinate = np.concatenate([list_coordinate[:, :2], list_coordinate[:, 3:5]],
                                                 axis=1)
            delta_x = list_coordinate[:, 0] - list_coordinate[:, 2]
            delta_y = list_coordinate[:, 1] - list_coordinate[:, 3]

        vectors = np.array([[delta_x[i], delta_y[i]] for i in range(len(delta_x))])
        if len(np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y <= 0)])==1:
            north_ = [np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y <= 0)][0]]
            east_, south_, west_ = [], [], []
            for i in range(len(vectors)):
                if i == north_[0]:
                    continue
                sim = vectors[i].dot(vectors[north_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[north_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[north_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    east_.append(i)
                elif theta >= 135 and theta < 225:
                    south_.append(i)
                elif theta >= 225 and theta < 315:
                    west_.append(i)
                else:
                    north_.append(i)
        elif len(np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x <= 0)]) == 1:
            east_ = np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x <= 0)]
            north_, south_, west_ = [], [], []
            for i in range(len(vectors)):
                if i == east_[0]:
                    continue
                sim = vectors[i].dot(vectors[east_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[east_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[east_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    south_.append(i)
                elif theta >= 135 and theta < 225:
                    west_.append(i)
                elif theta >= 225 and theta < 315:
                    north_.append(i)
                else:
                    east_.append(i)
        elif len(np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y > 0)]) == 1:
            south_ = np.arange(len(list_coordinate))[(abs(delta_y) > abs(delta_x)) & (delta_y > 0)]
            north_, east_, west_ = [], [], []
            for i in range(len(vectors)):
                if i == south_[0]:
                    continue
                sim = vectors[i].dot(vectors[south_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[south_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[south_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    west_.append(i)
                elif theta >= 135 and theta < 225:
                    north_.append(i)
                elif theta >= 225 and theta < 315:
                    east_.append(i)
                else:
                    south_.append(i)
        elif len(np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x > 0)]) == 1:
            west_ = np.arange(len(list_coordinate))[(abs(delta_x) > abs(delta_y)) & (delta_x > 0)]
            north_, east_, south_ = [], [], []
            for i in range(len(vectors)):
                if i == west_[0]:
                    continue
                sim = vectors[i].dot(vectors[west_[0]]) / (
                        np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[west_[0]]))
                theta = np.degrees(np.arccos(sim))
                crs = np.cross(vectors[i], vectors[west_[0]])
                if crs < 0:
                    theta = 360 - theta
                if theta < 135 and theta >= 45:
                    north_.append(i)
                elif theta >= 135 and theta < 225:
                    east_.append(i)
                elif theta >= 225 and theta < 315:
                    south_.append(i)
                else:
                    west_.append(i)
        else:
            return False, None
        if len(east_) >= 2 or len(south_) >= 2 or len(north_) >= 2 or len(west_) >= 2:
            return False, None
        east = east_[0] if len(east_) > 0 else -1
        west = west_[0] if len(west_) > 0 else -1
        south = south_[0] if len(south_) > 0 else -1
        north = north_[0] if len(north_) > 0 else -1
        list_coord_sort = [north, west, south, east]  # seq index

        return True, list_coord_sort

    def getCurrentPhase(self):
        """Return duration(list) and phase(list).
        Length of duration and phase is equal to number of phases in a cycle.
        Each element of a phase is a list, which contains 8 elements,
        respectively representing whether each entry lane of a standard intersection can pass.
        """
        index = self._env.sim.trafficlight.getProgram(self._id)
        logic = self._env.sim.trafficlight.getCompleteRedYellowGreenDefinition(self._id)[1]
        assert logic.programID == index
        assert len(logic.phases) % 2 == 0
        duration = []
        strs = []
        for i in range(0, len(logic.phases), 2):
            duration.append(
                logic.phases[i].duration + logic.phases[i+1].duration
            )
            strs.append(logic.phases[i].state)

        current_phase = []
        for current_phase_ in strs:
            tmp_ = []
            for i in range(len(self.entering_sequence_NWSE)):
                if self.entering_sequence_NWSE[i] != -1:
                    seq_i = self.entering_sequence[self.entering_sequence_NWSE[i]]
                    for s in self.seq:
                        if s in self.entering_approaches2phase[seq_i]:
                            res = 0 if s == 'l' else 1
                            tmp = 0
                            for kk in self.phase_index[2 * i + res]:
                                if current_phase_[kk] in ['g', "G"]:
                                    tmp = 1
                                elif current_phase_[kk] == 'r':
                                    tmp = 0
                                    break
                            tmp_.append(tmp)
                        else:
                            tmp_.append(0)
                else:
                    for _ in self.seq:
                        tmp_.append(0)
            current_phase.append(tmp_)
        return duration, current_phase

    def getCurrentPhaseDuration(self):
        return self._env.sim.trafficlight.getPhaseDuration(self._id)

    def set_interval_spillback(self, spillback):
        self.last_spillback = spillback

    def set_interval_flow(self, flow):
        self.last_flow = flow

    def get_interval_flow(self):
        return self.last_flow

    def set_interval_queue_length(self, queue_length):
        self.last_queue_length = queue_length

    def get_history_action(self, last_num=5):
        return self.history_duration[-last_num:], self.history_phases[-last_num:]

    def get_green_utilization(self, flow, duration, phase):
        total_interval = 60 * 15
        each_phase_time = np.zeros_like(duration)
        each_phase_duration = np.array(duration) * total_interval // sum(duration)
        each_phase_time += (total_interval // sum(duration))
        rest_time = total_interval % sum(duration)
        for i in range(len(duration)):
            if duration[i] > rest_time:
                each_phase_duration[i] += rest_time
                each_phase_time[i] += rest_time / duration[i]
                break
            else:
                rest_time -= duration[i]
                each_phase_duration[i] += duration[i]
                each_phase_time[i] += 1
        green_utilization = np.sum(np.array(flow) * np.array(phase), 1) / 2 * 2.5
        for i in range(len(green_utilization)):
            if green_utilization[i] <= self.single_min_phase_duration * each_phase_time[i]:
                green_utilization[i] = \
                    self.single_min_phase_duration * each_phase_time[i] / (each_phase_duration[i] + 1e-7)
            elif green_utilization[i] > each_phase_duration[i]:
                green_utilization[i] = 1
            else:
                green_utilization[i] = green_utilization[i] / (each_phase_duration[i] + 1e-7)
        self.last_green_utilization = green_utilization
        return green_utilization

    def get_road_capacity(self, last_num=4):
        # 考虑有信号灯的道路情况，最大通行能力与上一周期道路被分配到
        # 绿灯时长有关，与车辆通过路口的时长有关一般为2-3s

        #读取历史的方案,取last_num个周期，若1小时则=4，若单个周期则=1
        history_duration, history_phase = self.get_history_action(last_num)
        #初始化道路的绿灯时长
        road_green_time = np.zeros(len(history_phase[0][0]))
        for nums in range(len(history_duration)):
            #获取每一个历史方案
            phase_duration = history_duration[nums]
            phase_ = history_phase[nums]
            road_green_time += np.sum(np.array(phase_) * np.array(phase_duration).reshape(-1,1), axis = 0)

        #计算每条道路道路通行能力(一个小时)
        road_capacity = (road_green_time / (self.car_passtime)) * (last_num / len(history_duration))
        return road_capacity.tolist()

        # 以下为没有信号灯控制的道路的最大通行能力l
        # default in 'Krauss'-Model
        # minGap = 2.5
        # tau = 1
        # length = 5

        # road_capacity_dict = {}
        # for lane in self._incoming_lanes + self._outgoing_lanes:
        #     lane_max_speed = self._env.sim.lane.getMaxSpeed(lane)
        #     grossTimeHeadway = (length + minGap) / lane_max_speed + tau
        #     road_capacity_dict[lane] = 3600 / grossTimeHeadway
        # return road_capacity_dict
        # pass

    def get_spillback(self):
        # 目前实现的方式是读取道路的Occupancy，若大于某个阈值则认为发生spillback
        # 获取每个进口道路对应的出口道路，读取出口道路的Occupancy
        spillback_list = [0 for _ in range(len(self.phase_index))]
        ControlledLinks = self._env.sim.trafficlight.getControlledLinks(self._id)
        for phase in self.phase_index:
            for index in phase:
                out_road_id = ControlledLinks[index][0][1]
                occupancy = self._env.sim.lane.getLastStepOccupancy(out_road_id)
                haltingcar_ratio = \
                    self._env.sim.lane.getLastStepHaltingNumber(out_road_id) / \
                    (self._env.sim.lane.getLastStepVehicleNumber(out_road_id) + 1e-6)
                if occupancy > self.spillback_threshold and \
                        haltingcar_ratio > self.halting_threshold:
                    spillback_list[self.phase_index.index(phase)] = 1
                    break

        return spillback_list

    def get_phase_balance(self, green_utilization):

        # 每个相位的绿灯利用率/几个相位的绿灯利用率的加和
        balance = np.array([i/(np.sum(green_utilization)+1e-7) for i in green_utilization])
        return balance

    def reset_tl_logic(self):
        # 1. 生成随机的循环的相位（东西直行 A 、东西左转 D 、南北直行 E 、南北左转 H）并应用
        # 注意 ： 两个直行 绿灯时长50，两个左转40秒 这里设置两个类内变量进行控制
        #         self.through_green_time, self.left_green_time = 50, 40
        # 2. 返回不能选择的相位
        """
        Order: A: wt-et B: el-et C: wl-wt D: el-wl E: nt-st F: sl-st G: nt-nl H: nl-sl
        """
        logic = []
        unava_index = []
        self.right_index = []
        for i in range(len(self.lans_link)):
            if i not in np.concatenate(np.array(self.phase_index)):
                self.right_index.append(i)
        A = self.phase_index[3] + self.phase_index[7] # 东西直行
        B = self.phase_index[6] + self.phase_index[7] # 东
        C = self.phase_index[3] + self.phase_index[2] # 西
        D = self.phase_index[6] + self.phase_index[2] # 东西左转
        E = self.phase_index[1] + self.phase_index[5] # 南北直行
        F = self.phase_index[4] + self.phase_index[5] # 南
        G = self.phase_index[1] + self.phase_index[0] # 北
        H = self.phase_index[4] + self.phase_index[0] # 南北左转
        all_candidate = [A, B, C, D, E, F, G, H]
        index2dirc = {}
        for _, v in self._incominglanes_links.items():
            for _, pair in v.items():
                index2dirc[pair[0]] = pair[1]

        # random_type = np.random.choice(3)
        random_type = 0
        if random_type==0:
            self.sequence_ = [0, 3, 4, 7]
        elif random_type == 1:
            self.sequence_ = [4, 7, 0, 3]
        else:
            self.sequence_ = [1, 6, 2, 5]
            random.shuffle(self.sequence_) ### 随机生成

        self.sequence = []
        for index in self.sequence_:
            self.sequence.append(all_candidate[index])

        num_phase = 0
        self.masked_phase = []
        for i in range(len(self.sequence)):
            if len(self.sequence[i]) != 0:
                num_phase += 1
                self.masked_phase.append(0)
            else:
                self.masked_phase.append(1)
        self.max_phase_duration = self.single_max_phase_duration * num_phase

        if num_phase == 2:
            time = [25, 25, 25, 25]  # two of them is missed
        elif num_phase == 3:
            time = [23, 23, 23, 23]  # one of them is missed
        elif num_phase == 4:
            if random_type < 2:
                time = [20, 20, 20, 20]
            else:
                time = [23, 23, 23, 23]
        else:
            raise ValueError(self._id, num_phase)
        for i in range(len(self.sequence)):
            if len(self.sequence[i]) == 0:
                phase_ = list("r" * len(self.lans_link))
                for j in self.right_index:
                    phase_[j] = 'g'
                phase_ = self.getCurrentPhaseYellow("".join(phase_))
                logic.append([phase_, 0.1])
                logic.append([phase_, 0.1])
                unava_index.append(i)    # note: combined with sequence
                # unava_index.append(2*i+1)
            else:
                phase_ = list("r" * len(self.lans_link))  # ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', ...]
                for j in self.right_index:
                    phase_[j] = 'g'
                for j in self.sequence[i]:
                    if index2dirc[j] == 'r':
                        phase_[j] = 'g'
                    else:
                        phase_[j] = 'G'

                for a in self._incominglanes_links.keys():
                    for _, (index, dirt) in self._incominglanes_links[a].items():
                        if dirt == 'r' and phase_[index] != 'G':
                            phase_[index] = 'g'

                logic.append(["".join(phase_), time[i] - self.yellow_time])
                logic.append([self.getCurrentPhaseYellow("".join(phase_)), self.yellow_time])

        phases = []
        for i in logic:
            phases.append(
                self._env.sim.trafficlight.Phase(i[1], i[0], 0, 0))
        logic = self._env.sim.trafficlight.Logic("custom", 0, 0, phases)

        self._env.sim.trafficlight.setProgramLogic(self._id, logic)

        return unava_index

    def set_phase(self, action):
        """Set duration of a cycle phases."""
        index = self._env.sim.trafficlight.getProgram(self._id)
        logic = self._env.sim.trafficlight.getCompleteRedYellowGreenDefinition(self._id)[1]
        assert logic.programID == index
        assert len(logic.phases) % 2 == 0
        total_duration = 0
        for i in logic.phases:
            total_duration += i.duration

        if isinstance(action, list):
            assert len(action) == 4
            for i in range(4):
                if i not in self.unava_index and action[i] < 2:
                    action_type = action[i] % 2
                    if action_type == 0:
                        tmp = max(logic.phases[i * 2].duration - 5,
                                  self.single_min_phase_duration)
                        logic.phases[i * 2].duration = tmp
                    else:
                        tmp = min(5, abs(self.max_phase_duration - total_duration)) + logic.phases[
                            i * 2].duration
                        logic.phases[i * 2].duration = tmp
        else:
            if action == 8:
                return
            phase_index = int(action / 2)
            assert phase_index not in self.unava_index, "{}-{}-{}".format(self._id, action,
                                                                          self.unava_index)
            phase_index *= 2
            action_type = action % 2
            if action_type == 0:
                tmp = min(5, abs(self.max_phase_duration - total_duration)) + logic.phases[
                    phase_index * 2].duration
                logic.phases[phase_index * 2].duration = tmp
            else:
                tmp = max(logic.phases[phase_index * 2].duration - 5, self.single_min_phase_duration)
                logic.phases[phase_index * 2].duration = tmp
        self._env.sim.trafficlight.setProgramLogic(self._id, logic)
        # self._env.sim.trafficlight.setProgram(self._id, "custom")

    def getCurrentPhaseYellow(self, currentPhase):
        yellow_phase = list(currentPhase.replace("G", "y").replace("g", "y"))
        for a in self._incominglanes_links.keys():
            # if len(self._incominglanes_links[a].keys()) == 1 and list(self._incominglanes_links[a].values())[0][1] == 'r':
            #     yellow_phase[list(self._incominglanes_links[a].values())[0][0]] = 'g'
            for _, (index, dirt) in self._incominglanes_links[a].items():
                if dirt == 'r':
                    yellow_phase[index] = 'g'

        return "".join(yellow_phase)

    def update_timestep(self):
        self._previous_lane_vehicle_dict = self._lane_vehicle_dict.copy()
        self._update_lane_vehicle_info()

    def get_lane_traffic_volumn(self):
        traffic_volumn_dict = {}
        now_all_incoming_lanes_veh = set()
        for lane in self._incoming_lanes:
            now_all_incoming_lanes_veh.update(self._lane_vehicle_dict[lane])
        for lane in self._incoming_lanes:
            traffic_volumn = 0
            if lane in self._previous_lane_vehicle_dict:
                for veh in self._previous_lane_vehicle_dict[lane]:
                    if veh not in now_all_incoming_lanes_veh:
                        traffic_volumn += 1
            traffic_volumn_dict[lane] = traffic_volumn
        return traffic_volumn_dict

    def get_lane_queue_len(self):
        queue_len_dict = {}
        for lane in self._incoming_lanes:
            queue_len = self._env.sim.lanearea.getJamLengthMeters(lane)
            queue_len_dict[lane] = queue_len
        return queue_len_dict

    def _update_lane_vehicle_info(self) -> None:
        for lane in self._incoming_lanes + self._outgoing_lanes:
            self._lane_vehicle_dict[lane] = self._env.sim.lane.getLastStepVehicleIDs(lane)
            for veh in self._lane_vehicle_dict[lane]:
                if veh not in self._env.vehicle_info:
                    self._env.vehicle_info[veh] = {}

    def get_lane_car_number(self):
        car_number_dict = {}
        for lane in self._incoming_lanes:
            # car_number = self._env.sim.lanearea.getLastStepHaltingNumber(lane)
            # car_number = self._env.sim.lanearea.getLastStepVehicleNumber(lane) * 5
            car_number = self._env.sim.lane.getLastStepHaltingNumber(lane) * 5
            car_number_dict[lane] = car_number
        return car_number_dict

    def get_pressure(self):
        pressure = 0
        for lane in self._incoming_lanes:
            pressure += self._env.sim.lane.getLastStepHaltingNumber(lane)
        for lane in self._outgoing_lanes:
            pressure -= self._env.sim.lane.getLastStepHaltingNumber(lane)
        return abs(pressure)

    def get_lane_delay_time(self):
        delay_time_dict = {}
        for lane in self._incoming_lanes + self._outgoing_lanes:
            delay_time = 0
            for veh in self._lane_vehicle_dict[lane]:
                cur_distace = self._env.sim.vehicle.getDistance(veh)
                cur_time = self._env.sim.vehicle.getLastActionTime(veh)
                if 'distance' not in self._env.vehicle_info[veh]:
                    self._env.vehicle_info[veh]['distance'] = cur_distace
                    self._env.vehicle_info[veh]['time'] = cur_time
                else:
                    real_distance = cur_distace - self._env.vehicle_info[veh]['distance']
                    target_speed = self._env.sim.vehicle.getMaxSpeed(veh)
                    self._env.vehicle_info[veh]['distance'] = cur_distace
                    target_distance = (cur_time - self._env.vehicle_info[veh][
                        'time']) * target_speed
                    self._env.vehicle_info[veh]['time'] = cur_time
                    delay_time += (target_distance - real_distance) / (target_speed + 1e-8)
            delay_time_dict[lane] = delay_time
        return delay_time_dict

    def get_reward(self, reward_type):  # TODO add bonus
        """flow, green_utilization, green_balance"""
        reward = {}
        if 'green_utilization' in reward_type:
            reward['green_utilization'] = np.sum(self.last_green_utilization * (1 - np.array(self.masked_phase))) / \
                                          np.sum(1 - np.array(self.masked_phase)) * 2
        if 'green_balance' in reward_type:
            gb = self.get_phase_balance(self.last_green_utilization)
            reward['green_balance'] = -np.abs(gb - 1 / (len(self.last_green_utilization) - sum(self.masked_phase))).sum()
        if 'flow' in reward_type:
            # _flow = self.last_flow
            # _flow_ = _flow[0]
            # for i in range(1, len(_flow)):
            #     for k in _flow[i]:
            #         _flow_[k] += _flow[i][k]
            # reward['flow'] = sum(_flow_.values()) / len(_flow_) / 25    #  rescale

            _flow = copy.deepcopy(self.last_flow)
            _flow_ = _flow[0]
            flow_tmp = []
            for i in range(len(self.entering_sequence_NWSE)):
                if self.entering_sequence_NWSE[i] != -1:
                    seq_i = self.entering_sequence[self.entering_sequence_NWSE[i]]
                    for s in self.seq:
                        if s in self.entering_approaches2phase[seq_i]:
                            flow = 0
                            for l in self.entering_approaches2phase[seq_i][s]:
                                flow = max(flow, _flow_[l[0]])
                            flow_tmp.append(flow)
                        else:
                            flow_tmp.append(0)
                else:
                    for _ in self.seq:
                        flow_tmp.append(0)
            flow_phase = np.array([flow_tmp[3] + flow_tmp[7], flow_tmp[6] + flow_tmp[2],
                                   flow_tmp[1] + flow_tmp[5], flow_tmp[4] + flow_tmp[0]])
            reward['flow'] = sum(flow_phase) / len(flow_phase) / 25

        if 'queue_len' in reward_type:

            _flow = copy.deepcopy(self.last_flow)
            _flow_ = _flow[0]
            flow_tmp = []
            for i in range(len(self.entering_sequence_NWSE)):
                if self.entering_sequence_NWSE[i] != -1:
                    seq_i = self.entering_sequence[self.entering_sequence_NWSE[i]]
                    for s in self.seq:
                        if s in self.entering_approaches2phase[seq_i]:
                            flow = 0
                            for l in self.entering_approaches2phase[seq_i][s]:
                                flow = max(flow, _flow_[l[0]])
                            flow_tmp.append(flow)
                        else:
                            flow_tmp.append(0)
                else:
                    for _ in self.seq:
                        flow_tmp.append(0)
            flow_phase = np.array([flow_tmp[3] + flow_tmp[7], flow_tmp[6] + flow_tmp[2],
                                   flow_tmp[1] + flow_tmp[5], flow_tmp[4] + flow_tmp[0]])
            flow_phase = flow_phase/sum(flow_phase)

            _queue_len = self.last_queue_length
            _queue_len_ = _queue_len[0]

            # for i in range(1, len(_queue_len)):
            #     for k in _queue_len[i]:
            #         _queue_len_[k] += _queue_len[i][k]
            #
            # reward['queue_len'] = -sum(_queue_len_.values()) / len(_queue_len_) / 75

            _queue_len_tmp = []
            for i in range(len(self.entering_sequence_NWSE)):
                if self.entering_sequence_NWSE[i] != -1:
                    seq_i = self.entering_sequence[self.entering_sequence_NWSE[i]]
                    for s in self.seq:
                        if s in self.entering_approaches2phase[seq_i]:
                            queue_len = 0
                            for l in self.entering_approaches2phase[seq_i][s]:
                                queue_len = max(queue_len, _queue_len_[l[0]])
                            _queue_len_tmp.append(queue_len)
                        else:
                            _queue_len_tmp.append(0)
                else:
                    for _ in self.seq:
                        _queue_len_tmp.append(0)
            queue_len_phase = np.array([_queue_len_tmp[3] + _queue_len_tmp[7], _queue_len_tmp[6] + _queue_len_tmp[2],
                                   _queue_len_tmp[1] + _queue_len_tmp[5], _queue_len_tmp[4] + _queue_len_tmp[0]])

            # reward['queue_len'] = -sum(_queue_len_tmp) / len(_queue_len_tmp) / 50
            reward['queue_len'] = -sum(queue_len_phase * flow_phase) / 1000
        return reward

    def get_state(self):
        # main_direction_lanes = []
        mask = []
        duration, current_phase = self.getCurrentPhase()
        self.history_phases.append(current_phase)
        self.history_duration.append(duration)
        history_duration, history_phase = self.get_history_action()
        _flow = self.last_flow
        _queue = self.last_queue_length
        _queue_ = _queue[0]
        _flow_ = _flow[0]
        for i in range(1, len(_flow)):
            for k in _flow[i]:
                _flow_[k] += _flow[i][k]
                _queue_[k] += _queue[i][k]


        spillback = np.mean(self.last_spillback, 0)
        road_capacity = self.get_road_capacity()
        flow_tmp = []  #  todo 三口道验证
        queue_tmp = []
        for i in range(len(self.entering_sequence_NWSE)):
            if self.entering_sequence_NWSE[i] != -1:
                seq_i = self.entering_sequence[self.entering_sequence_NWSE[i]]
                for s in self.seq:
                    # obs = {}
                    if s in self.entering_approaches2phase[seq_i]:
                        flow = 0
                        queue = 0
                        for l in self.entering_approaches2phase[seq_i][s]:
                            flow = max(flow, _flow_[l[0]])
                            queue = max(queue, _queue_[l[0]])
                            # flow += _flow_[l[0]]
                        # if "flow" in self.state:
                        #     flow_tmp.append(flow / len(self.entering_approaches2phase[seq_i][s]))
                        flow_tmp.append(flow)
                        queue_tmp.append(queue)
                        #main_direction_lanes.append(obs)
                        mask.append(0)
                    else:
                        mask.append(1)
                        #main_direction_lanes.append({})
                        flow_tmp.append(0)
                        queue_tmp.append(0)
            else:
                for _ in self.seq:
                    mask.append(1)
                    flow_tmp.append(0)
                    queue_tmp.append(0)
                    # main_direction_lanes.append({})

        green_utilization = self.get_green_utilization(flow_tmp, duration, current_phase)
        green_balance = self.get_phase_balance(green_utilization)

        obs = {
            'flow': flow_tmp,
            'spillback': spillback,
            'road_capacity': road_capacity,
            'history_duration': history_duration,
            'history_phase': history_phase,
            'duration': duration,
            'current_phase': current_phase,
            'green_utilization': green_utilization,
            'green_balance': green_balance,
            'mask': mask,
            'masked_phase': self.masked_phase,
            'phase_index': copy.deepcopy(self.sequence_),
            'queue_length': queue_tmp
        }

        return obs
