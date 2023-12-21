import math

import numpy as np


class Node:
    def __init__(self, id, pos, E_n, P_n, bandwidth, type, w, C_n, config):
        self.id = id
        self.position = pos
        self.E_n = E_n  # J/cycle--the CPU energy consumption to implement one cycle at node ut
        self.P_n = P_n  # w--the transmit power of node ut with Pmax,ut being the transmit power budget of the node
        self.bandwidth = bandwidth  # MHz
        self.type = type
        self.w = w  # cycle/bit--the number of CPU cycles required to compute one bit (from gpt suggest)
        self.C_n = C_n  # cycle/s --the available computation resource, i.e., in cycle/s, of node
        self.config = config

    def offloading_time(self, data_size_on_local, data_size_on_remote, target_node):
        computation_delay = (data_size_on_local * self.w) / self.C_n
        offloading_delay = data_size_on_remote / self.get_transmission_rate(target_node)
        return max(offloading_delay, computation_delay)

    def target_node_offloading_time(self, data_size_on_local, data_size_on_remote, target_node):
        computation_delay = (data_size_on_local * target_node.w) / target_node.C_n
        offloading_delay = data_size_on_remote / self.get_transmission_rate(target_node)
        # print("---:", target_node.type, target_node.id, " com_delay:", computation_delay,
        #       " off_delay:", offloading_delay, " node_dis:", self.get_dis(self.position, target_node.position),
        #       " rate:", self.get_transmission_rate(target_node), " bandwidth:", self.bandwidth, " loss:", self.get_path_loss(target_node))
        return max(offloading_delay, computation_delay)

    def los_probability_U2V(self, target_position):
        """
        计算车辆与无人机之间的LoS概率。
        :param target_position:
        """
        uav_position = self.position if self.type == "uav" else target_position
        vehicle_position = self.position if self.type == "vehicle" else target_position
        d_vu = self.get_dis(uav_position, vehicle_position)

        # 计算仰角
        y_u = uav_position[1]
        elevation_angle = np.arcsin(y_u / d_vu)

        # 使用给定的公式计算LoS概率
        probability = 1 / (1 + self.config['communication_config']["eta1"] * np.exp(
            -self.config['communication_config']["eta2"] * (
                    elevation_angle - self.config['communication_config']["eta1"])))
        return probability

    def path_loss_U2V(self, target_position):
        # 光速v_c，单位：m/s
        v_c = 3 * 10 ** 8

        # 计算LoS和NLoS的概率
        uav_position = self.position if self.type == "uav" else target_position
        vehicle_position = self.position if self.type == "vehicle" else target_position
        h_LoS = self.los_probability_U2V(target_position)
        h_NLoS = 1 - h_LoS
        d_vu = self.get_dis(uav_position, vehicle_position)
        # 计算自由空间路径损耗L^FS
        L_FS = 20 * np.log10(d_vu) + 20 * np.log10(self.config['communication_config']["fc"]) + 20 * np.log10(
            4 * np.pi / v_c)
        # 计算LoS和NLoS情况下的路径损耗
        L_LoS = L_FS + self.config['communication_config']["eta_LoS"]
        L_NLoS = L_FS + self.config['communication_config']["eta_NLoS"]
        # 计算总路径损耗
        L_total = h_LoS * L_LoS + h_NLoS * L_NLoS
        return L_total

    def path_loss_V2V(self, target_position, zeta_mode='reverse'):
        """
        计算城市环境中车辆之间的路径损耗。this model excerpted from <Path Loss Modeling for Vehicle-to-Vehicle Communications>
        :param target_position:
        :param zeta_mode: 路径损耗模式（'reverse', 'forward', 'convoy'）
        :return: 路径损耗
        """

        # 计算两车之间的距离
        d_vv = self.get_dis(self.position, target_position)
        d_vv = d_vv if d_vv>=self.config['communication_config']['d0'] else self.config['communication_config']['d0']
        # 计算正态随机分布变量 绝大多数值（约 99.7%）将落在 [-5.1, 5.1] 的范围内（5.1 = 3 * 1.7）
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
        L_vv = self.config['communication_config']["L0vv"] + 10 * self.config['communication_config'][
            "eta3"] * np.log10(d_vv / self.config['communication_config']["d0"]) + \
               X_eta4 + zeta * self.config['communication_config']["Lcvv"]
        return L_vv

    def path_loss_U2U(self, target_position):
        """
        :param target_position:
        :return: 路径损耗
        """
        v_c = 3 * 10 ** 8  # 光速
        d_uu = self.get_dis(self.position, target_position)
        L_uu = 20 * np.log10(d_uu) + 20 * np.log10(self.config['communication_config']["fc"]) + 20 * np.log10(
            4 * np.pi / v_c)
        return L_uu

    def get_path_loss(self, target_node):
        # print('this id',self.id,'target id :',target_node.id)
        loss = 0
        if self.type == "uav" and target_node.type == "uav":
            return self.path_loss_U2U(target_node.position)

        elif (self.type == "uav" and target_node.type == "vehicle") or (
                self.type == "vehicle" and target_node.type == "uav"):
            return self.path_loss_U2V(target_node.position)
        else:
            return self.path_loss_V2V(target_node.position)

    def energy_consumption_of_node_transmission(self, data_size, target_node):
        """
        计算传输数据造成的能耗
        :return:传输能耗
        """
        transmission_rate = self.get_transmission_rate(target_node)
        return (data_size * self.P_n) / transmission_rate

    def energy_consumption_of_node_computation(self, data_size):
        return data_size * self.w * self.E_n

    def get_transmission_rate(self, target_node):
        """
        get the transmission rate
        :param target_node:
        :return: the transmission rate b/s
        """

        loss = self.get_path_loss(target_node)
        return self.bandwidth * math.log2(
            1 + self.P_n / (self.config['communication_config']['p_noise'] * 10 ** (loss / 10)))

    def get_dis(self, position1, position2):
        """
        :param position1:位置1
        :param position2:位置2
        :return: 两个坐标的欧式距离
        """
        return np.sqrt((position1[0] - position2[0]) ** 2 +
                       (position1[1] - position2[1]) ** 2 +
                       (position1[2] - position2[2]) ** 2)

class UAV(Node):
    def __init__(self, config, id):
        super(UAV, self).__init__(id=id, pos=config['uav_config']['pos'][id], E_n=config['uav_config']['E_n'],
                                  P_n=config['uav_config']['P_n'], bandwidth=config['uav_config']['bandwidth'],
                                  type=config['uav_config']['type'], w=config['uav_config']['w'],
                                  C_n=config['uav_config']['C_n'], config=config)

    def node_is_in_range(self, node):
        if node.type == 'uav':
            return True
        if (node.position[0] <= self.position[0] + 250) and (node.position[0] >= self.position[0] - 250) \
                and (node.position[2] <= self.position[2] + 250) and (node.position[2] >= self.position[2] - 250):
            return True
        return False


class Vehicle(Node):
    def __init__(self, config, id, path, time_line, uav_len):
        super(Vehicle, self).__init__(id=id,
                                      pos=[0, 0, 0],
                                      E_n=config['vehicle_config']['E_n'][id - uav_len],
                                      P_n=config['vehicle_config']['P_n'][
                                          (id - uav_len) % len(config['vehicle_config']['P_n'])],
                                      bandwidth=config['vehicle_config']['bandwidth'],
                                      type=config['vehicle_config']['type'],
                                      w=config['vehicle_config']['w'],
                                      C_n=config['vehicle_config']['C_n'][id - uav_len],
                                      config=config)
        self.path = path
        self.position = self.path[time_line]

    def node_is_in_range(self, node):
        if node.type == 'uav' \
                and node.position[0] + 250 >= self.position[0] >= node.position[0] - 250 \
                and node.position[2] + 250 >= self.position[2] >= node.position[2] - 250:
            return True
        if self.get_dis(self.position, node.position) <= self.config['vehicle_config']['vehicle_communication_range']:
            return True
        return False

    def reset(self, time_line):
        self.position = self.path[time_line]

    def run_step(self, time_line):
        self.position = self.path[time_line]
