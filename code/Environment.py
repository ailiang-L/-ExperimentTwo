import gym
from gym import spaces
import numpy as np
from Node import Node
from PathCreator import PathCreator
import yaml


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
            self.nodes.append(Node())

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

    def los_probability_U2V(self, uav_position, vehicle_position):
        """
        计算车辆与无人机之间的LoS概率。

        :param uav_position: 无人机的位置，格式为 (x_u, y_u, z_u)
        :param vehicle_position: 车辆的位置，格式为 (x_v, y_v, z_v)
        :param eta1, eta2: 信道传播环境决定的常量值
        :return: LoS概率
        """

        # 解析无人机和车辆的位置
        x_u, y_u, z_u = uav_position
        x_v, y_v, z_v = vehicle_position

        # 计算欧氏距离
        d_vu = np.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2 + (z_u - z_v) ** 2)

        # 计算仰角
        elevation_angle = np.arcsin(y_u / d_vu)

        # 使用给定的公式计算LoS概率
        probability = 1 / (1 + self.config['communication_config']["eta1"] * np.exp(
            -self.config['communication_config']["eta2"] * (elevation_angle - self.config['communication_config']["eta1"])))

        return probability

        # 定义路径损耗函数

    def path_loss_U2V(self, uav_position, vehicle_position):
        # 光速v_c，单位：m/s
        v_c = 3 * 10 ** 8

        # 计算LoS和NLoS的概率
        h_LoS = self.los_probability_U2V(uav_position, vehicle_position)
        h_NLoS = 1 - h_LoS

        # 解析无人机和车辆的位置来计算距离d_vu
        x_u, y_u, z_u = uav_position
        x_v, y_v, z_v = vehicle_position
        d_vu = np.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2 + (z_u - z_v) ** 2)

        # 计算自由空间路径损耗L^FS
        L_FS = 20 * np.log10(d_vu) + 20 * np.log10(self.config['communication_config']["fc"]) + 20 * np.log10(4 * np.pi / v_c)

        # 计算LoS和NLoS情况下的路径损耗
        L_LoS = L_FS + self.config['communication_config']["eta_LoS"]
        L_NLoS = L_FS + self.config['communication_config']["eta_NLoS"]

        # 计算总路径损耗
        L_total = h_LoS * L_LoS + h_NLoS * L_NLoS

        return L_total

    def path_loss_V2V(self, vehicle1_position, vehicle2_position, zeta_mode):
        """
        计算城市环境中车辆之间的路径损耗。

        :param vehicle1_position: 发送者车辆的位置
        :param vehicle2_position: 接收者车辆的位置
        :param zeta_mode: 路径损耗模式（'reverse', 'forward', 'convoy'）
        :return: 路径损耗
        """

        # 计算两车之间的距离
        d_vv = self.get_dis(vehicle1_position, vehicle2_position)

        # 计算正态随机分布变量
        X_eta4 = np.random.normal(0, self.config['communication_config']["eta4"])

        # 根据zeta_mode来决定zeta的值
        if zeta_mode == 'reverse':
            zeta = 1
        elif zeta_mode == 'forward':
            zeta = -1
        elif zeta_mode == 'convoy':
            zeta = 0
        else:
            raise ValueError("Invalid zeta_mode. Choose from 'reverse', 'forward', or 'convoy'.")

        # 使用给定的公式计算路径损耗
        L_vv = self.config['communication_config']["L0vv"] + 10 * self.config['communication_config']["eta3"] * np.log10(
            d_vv / self.config['communication_config']["d0"]) + X_eta4 + zeta * self.config['communication_config']["Lcvv"]

        return L_vv

    def path_loss_U2U(self, drone1_position, drone2_position):
        """
        计算两个无人机之间的路径损耗。

        :param drone1_position: 无人机1的位置 (x, y, z)
        :param drone2_position: 无人机2的位置 (x, y, z)
        :param fc: 载波频率
        :param vc: 光速 (默认为3e8 m/s)

        :return: 路径损耗
        """
        v_c = 3 * 10 ** 8
        # 计算两个无人机之间的欧式距离
        d_uu = np.sqrt((drone1_position[0] - drone2_position[0]) ** 2 +
                       (drone1_position[1] - drone2_position[1]) ** 2 +
                       (drone1_position[2] - drone2_position[2]) ** 2)

        # 使用给定的公式计算路径损耗
        L_uu = 20 * np.log10(d_uu) + 20 * np.log10(self.config['communication_config']["fc"]) + 20 * np.log10(4 * np.pi / v_c)
        return L_uu

    def energy_consumption_of_node_transimission(self, node1, node2, data_size):
        """
        计算传输数据造成的能耗
        :param P_n: 节点的发送功率
        :return:传输能耗
        """
        P_n = node1.P_n
        transimission_rate = self.get_transimisssion_rate(node1, node2)
        return (data_size * P_n) / transimission_rate

    def get_transimission_rate(self, node1, node2):
        # dis=self.get_dis(node1.position,node2.position)
        loss = 0
        bandwidth = node1.bandwidth  # 带宽: 1 MHz
        P_n = node1.P_n
        if (node1.type == "UAV" and node2.type == "UAV"):
            loss = self.path_loss_U2U(node1.position, node2.position)
        elif ((node1.type == "UAV" and node2.type == "vehicle") or (node1.type == "vehicle" and node2.type == "UAV")):
            loss = self.path_loss_U2V(node1.position, node2.position)
        else:
            loss = self.path_loss_V2V(node1.position, node2.position)
        return bandwidth * np.log2(1 + (P_n / self.config['communication_config']["P_noise"]) * loss)

    def get_dis(self, position1, position2):
        '''
        :param position1:位置1
        :param position2:位置2
        :return: 两个坐标的欧式距离
        '''
        if not isinstance(position1, np.ndarray):
            position1 = np.array(position1)
        if not isinstance(position2, np.ndarray):
            position2 = np.array(position2)

        return np.sqrt((position1[0] - position2[0]) ** 2 +
                       (position1[1] - position2[1]) ** 2 +
                       (position1[2] - position2[2]) ** 2)
