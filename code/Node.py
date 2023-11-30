class Node:
    def __init__(self, id, pos, E_n, P_n, bandwidth, type, w):
        self.id = id
        self.position = pos
        self.E_n = E_n  # J/cycle--the CPU energy consumption to implement one cycle at node ut
        self.P_n = P_n  # w--the transmit power of node ut with Pmax,ut being the transmit power budget of the node
        self.bandwidth = bandwidth  # MHz
        self.type = type
        self.w = w  # cycle/bit--the number of CPU cycles required to compute one bit (from gpt suggest)


class UAV(Node):
    def __init__(self, config, id):
        super(UAV, self).__init__(id, config['uav_config']['pos'][id], config['uav_config']['E_n'],
                                  config['uav_config']['P_n'], config['uav_config']['bandwidth'],
                                  config['uav_config']['type'], config['uav_config']['w'])

    def get_energy_cost(self, data_size):
        return data_size * self.w * self.E_n

    def offloading_time(self, node1, node2, data_size):
        """
        卸载时间包含了两个部分，一部分为计算时延，另外一部分为传输时延，但是卸载时延为二者的最大值
        :param node1:
        :param node2:
        :return:卸载时延
        """
        C_n = 0
        conputation_delay = data_size / C_n
        offloading_delay = data_size / self.get_transimisssion_rate(node1, node2)
        return max(offloading_delay, conputation_delay)


class Vehicle:
    def __init__(self, config, id):
        super(Vehicle, self).__init__(id, [0, 0, 0], config['vehicle_config']['E_n'],
                                  config['vehicle_config']['P_n'], config['vehicle_config']['bandwidth'],
                                  config['vehicle_config']['type'], config['vehicle_config']['w'])

    def get_energy_cost(self, data_size):
        return data_size * self.w * self.E_n
