import random

import gym
from gym import spaces
import numpy as np
from Node import Node
from PathCreator import PathCreator
import yaml
import sys
from Node import *
from LoadParameters import *


class OffloadingEnv(gym.Env):
    def __init__(self, trajectory_list):
        super(OffloadingEnv, self).__init__()
        # 加载配置文件
        self.config = load_parameters()
        # 设置随机种子
        random.seed(self.config['random_seed'])
        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=np.array([0, -np.inf, 0, 0, 0, 0, 0, 0]),
                                            high=np.array(
                                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(10)  # 动作空间大小为10

        self.trajectory_list = trajectory_list  # 车辆轨迹数据
        self.current_step = 0
        self.current_node = None
        self.target_node = None

        # 定义车辆的路径
        self.vehicle_paths = PathCreator(self.config['vehicle_config']["car_speed"],
                                         self.config['vehicle_config']["run_time"],
                                         self.config['vehicle_config']["time_slot"],
                                         self.config['vehicle_config']["path_num"],
                                         self.config['vehicle_config']["forward_probability"],
                                         self.config['vehicle_config']['random_seed'])
        # 时间线 随机一个时间点用于表示一个不确定的时间点进入任务计算状态
        self.time_line = random.randint(0, 1500)
        # 节点定义：4个无人机，20个车辆
        self.nodes = []
        # 定义无人机
        for i in len(self.config['uav_config']['pos']):
            self.nodes.append(UAV(self.config, i))
        # 定义车
        for i in range(self.config['vehicle_path_config']["vehicle_num"]):
            self.nodes.append(Vehicle(self.config, i, self.vehicle_paths[i], self.time_line))
        # 定义任务
        self.data_size = self.config['data_size']

    def step(self, action):
        # 执行一个时间步骤 TODO 时间线的更新有问题，不应该是直接+1
        self.time_line += 1
        # 选择下一个节点
        task_split_granularity = self.config['task_split_granularity'][action]
        data_size_on_local = int(task_split_granularity * self.data_size)
        data_size_on_remote = self.data_size - data_size_on_local
        # 计算奖励值
        reward = self.get_reward(self.current_node, self.target_node, data_size_on_local, data_size_on_remote)
        # 环境进入下一个状态
        self.current_node = self.target_node
        self.target_node = self.choose_target_node(self.current_node)
        # 节点切换以后对应数据大小也切换
        self.data_size = data_size_on_remote
        data_size_on_local = 0
        data_size_on_remote = 0
        # 构造下一个状态
        state = self.construct_state(self.current_node, self.target_node, self.data_size)
        # 检查是否为结束状态
        done = self.data_size == 0
        self.current_step += 1

        return state, reward, done, {}

    def reset(self):
        # 重置环境状态
        self.time_line = random.randint(0, 1500)
        self.data_size = self.config['data_size']
        self.current_step = 0
        # 重置车辆的位置
        for i in self.nodes:
            if i.type == 'vehicle':
                i.reset(self.time_line)
        # 随机选择节点以接收一个任务
        node_index = random.randint(0, len(self.nodes))
        self.current_node = self.nodes[node_index]
        self.target_node = self.choose_target_node(self.current_node)

        initial_state = self.construct_state(self.current_node, self.target_node, self.data_size)  # 初始化状态
        return initial_state

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def choose_target_node(self, current_node):
        min_value = sys.maxsize
        min_index = -1
        for i in range(len(self.nodes)):
            if current_node.id == self.nodes[i].id or current_node.node_is_in_range(self.nodes[i]) is False:
                continue
            e = self.nodes[i].energy_consumption_of_node_computation(
                1) + current_node.energy_consumption_of_node_transmission(1, self.nodes[i])
            t = current_node.offloading_time(1, 1, self.nodes[i])
            weight = self.config['node_choose_config']['e_weight'] * e + self.config['node_choose_config'][
                't_weight'] * t
            if weight < min_value:
                min_value = weight
                min_index = i
        assert self.nodes[min_index].id != current_node.id
        return self.nodes[min_index]

    def construct_state(self, current_node, target_node, data_size):
        s_t = current_node.w * data_size
        loss = current_node.get_path_loss(target_node)
        c_nt = current_node.C_n
        p_nt = current_node.P_n
        e_nt = current_node.E_n
        c_nt_next = target_node.C_n
        p_nt_next = target_node.P_n
        e_nt_next = target_node.E_n
        state = np.array([s_t, loss, c_nt, p_nt, e_nt, c_nt_next, p_nt_next, e_nt_next], dtype=np.float32)
        return state

    def get_reward(self, current_node, target_node, data_size_on_local, data_size_on_remote):
        e = current_node.energy_consumption_of_node_computation(
            data_size_on_local) + current_node.energy_consumption_of_node_transmission(data_size_on_remote, target_node)
        t = current_node.offloading_time(data_size_on_local, data_size_on_remote, current_node)
        return e * self.config['reward_config']['e_weight'] + t * ['reward_config']['t_weight']
