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

    def offloading_time(self, data_size, target_node):
        """
        卸载时间包含了两个部分，一部分为计算时延，另外一部分为传输时延，但是卸载时延为二者的最大值
        :param target_node:
        :param data_size:
        :return:卸载时延
        """
        computation_delay = (data_size * self.w) / self.C_n
        offloading_delay = data_size / self.get_transimisssion_rate(target_node.position)
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
        h_LoS = self.los_probability_U2V(uav_position, vehicle_position)
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

    def path_loss_V2V(self, target_position, zeta_mode='convoy'):
        """
        计算城市环境中车辆之间的路径损耗。this model excerpted from <Path Loss Modeling for Vehicle-to-Vehicle Communications>
        :param target_position:
        :param zeta_mode: 路径损耗模式（'reverse', 'forward', 'convoy'）
        :return: 路径损耗
        """

        # 计算两车之间的距离
        d_vv = self.get_dis(self.position, target_position)
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
        L_vv = self.config['communication_config']["L0vv"] + 10 * self.config['communication_config'][
            "eta3"] * np.log10(
            d_vv / self.config['communication_config']["d0"]) + X_eta4 + zeta * self.config['communication_config'][
                   "Lcvv"]

        return L_vv

    def path_loss_U2U(self, target_position):
        """
        :param target_position:
        :return: 路径损耗
        """
        v_c = 3 * 10 ** 8  # 光速
        # 计算两个无人机之间的欧式距离
        d_uu = self.get_dis(self.position, target_position)
        L_uu = 20 * np.log10(d_uu) + 20 * np.log10(self.config['communication_config']["fc"]) + 20 * np.log10(
            4 * np.pi / v_c)
        return L_uu

    def energy_consumption_of_node_transmission(self, data_size, target_node):
        """
        计算传输数据造成的能耗
        :return:传输能耗
        """
        transmission_rate = self.get_transmission_rate(target_node)
        return (data_size * self.P_n) / transmission_rate

    def get_transmission_rate(self, target_node):
        """
        get the transmission rate
        :param target_node:
        :return: the transmission rate
        """
        if self.type == "uav" and target_node.type == "uav":
            loss = self.path_loss_U2U(self.position, target_node.position)
        elif (self.type == "uav" and target_node.type == "vehicle") or (
                self.type == "vehicle" and target_node.type == "uav"):
            loss = self.path_loss_U2V(self.position, target_node.position)
        else:
            loss = self.path_loss_V2V(self.position, target_node.position)
        return self.bandwidth * np.log2(1 + (self.P_n / self.config['communication_config']["P_noise"]) * loss)????

    def get_dis(self, position1, position2):
        """
        :param position1:位置1
        :param position2:位置2
        :return: 两个坐标的欧式距离
        """
        if not isinstance(position1, np.ndarray):
            position1 = np.array(position1)
        if not isinstance(position2, np.ndarray):
            position2 = np.array(position2)

        return np.sqrt((position1[0] - position2[0]) ** 2 +
                       (position1[1] - position2[1]) ** 2 +
                       (position1[2] - position2[2]) ** 2)


class UAV(Node):
    def __init__(self, config, id):
        super(UAV, self).__init__(id, config['uav_config']['pos'][id], config['uav_config']['E_n'],
                                  config['uav_config']['P_n'], config['uav_config']['bandwidth'],
                                  config['uav_config']['type'], config['uav_config']['w'],
                                  config['uav_config']['C_n'], config)

    def get_energy_cost(self, data_size):
        return data_size * self.w * self.E_n


class Vehicle:
    def __init__(self, config, id):
        super(Vehicle, self).__init__(id, [0, 0, 0], config['vehicle_config']['E_n'],
                                      config['vehicle_config']['P_n'], config['vehicle_config']['bandwidth'],
                                      config['vehicle_config']['type'], config['vehicle_config']['w'],
                                      config['vehicle_config']['C_n'], config)

    def get_energy_cost(self, data_size):
        return data_size * self.w * self.E_n
