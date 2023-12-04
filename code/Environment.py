import gym
from gym import spaces
import numpy as np
from Node import Node
from PathCreator import PathCreator
import yaml
import sys


class OffloadingEnv(gym.Env):
    """
    自定义卸载任务的环境。
    """

    def __init__(self, trajectory_list):
        super(OffloadingEnv, self).__init__()
        # 加载配置文件
        with open('../config/parameters.yaml', 'r') as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=np.array([0, -np.inf, 0, 0, 0, 0, 0, 0]),
                                            high=np.array(
                                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(10)  # 动作空间大小为10

        self.trajectory_list = trajectory_list  # 车辆轨迹数据
        self.current_step = 0

        # 定义车辆的路径
        self.vehicle_paths = PathCreator(self.vehicle_config["car_speed"], self.vehicle_config["run_time"],
                                         self.vehicle_config["time_slot"], self.vehicle_config["path_num"],
                                         self.vehicle_config["forward_probability"])

        # 时间线
        self.time_line = 0.0
        # 节点定义：4个无人机，20个车辆
        self.nodes = []
        # 定义无人机
        for i in self.uav_config.keys():
            self.nodes.append(Node(i, "uav"))
        # 定义车
        for i in range(self.vehicle_config["vehicle_num"]):
            pass

    def step(self, action):
        # 执行一个时间步骤
        # ... 计算下一状态和奖励 ...

        # 示例状态（这需要根据您的特定场景来定制）
        state = np.random.random(8)
        reward = 0  # 奖励函数，您可以自定义

        done = self.current_step >= len(self.trajectory_list) - 1
        self.current_step += 1

        return state, reward, done, {}

    def reset(self):
        # 重置环境状态
        self.current_step = 0
        initial_state = np.random.random(8)  # 初始化状态
        return initial_state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # 打印出当前步骤的一些信息
        print(f"Step: {self.current_step}")

    def close(self):
        pass

    def choose_target_node(self, current_node):
        min_value = sys.maxint
        min_index = -1
        for i in range(len(self.nodes)):
            if current_node.id == self.nodes[i].id:
                continue
            e = self.nodes[i].energy_consumption_of_node_computation(
                1) + current_node.energy_consumption_of_node_transmission(1, self.nodes[i])
            t = current_node.offloading_time(1, 1, self.nodes[i])
            weight = self.config['node_choose_config']['e_weight'] * e + self.config['node_choose_config'][
                't_weight'] * t
            if weight < min_value:
                min_value = weight
                min_index = i
        assert self.nodes[i].id != current_node.id
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
