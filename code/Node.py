class Node:
    def __init__(self, pos=[0, 0, 0],E_n=7e-14,P_n=1,bandwidth=0,type="null",w=0):
        self.position = pos
        self.E_n = E_n  # J/cycle--the CPU energy consumption to implement one cycle at node ut
        self.P_n = P_n  # w--the transmit power of node ut with Pmax,ut being the transmit power budget of the node
        self.bandwidth = bandwidth  # MHz
        self.type = type
        self.w = w  # cycle/bit--the number of CPU cycles required to compute one bit (from gpt suggest)

class UAV(Node):
    def __init__(self, pos=[0, 0, 0],E_n=7e-14,P_n=1):
        self.position = pos
        self.E_n = E_n  # J/cycle--the CPU energy consumption to implement one cycle at node ut
        self.P_n = P_n  # w--the transmit power of node ut with Pmax,ut being the transmit power budget of the node
        self.bandwidth = 1  # MHz
        self.type = "uav"
        self.w = 15  # cycle/bit--the number of CPU cycles required to compute one bit (from gpt suggest)

    def get_energy_cost(self,data_size):
        return  data_size*self.w*self.E_n

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
    def __init__(self, pos=[0, 0, 0],E_n=14e-8,P_n=1):
        self.position = pos
        self.E_n = E_n  # J/cycle--the CPU energy consumption to implement one cycle at node ut
        self.P_n = P_n  # w--the transmit power of node ut with Pmax,ut being the transmit power budget of the node
        self.bandwidth = 1  # MHz
        self.type = "vehicle"
        # if the task size is 25e6 bit then the cycle is 0.25e9 cycle
        self.w=10 # cycle/bit--the number of CPU cycles required to compute one bit (from gpt suggest)

    def get_energy_cost(self,data_size):
        return  data_size*self.w*self.E_n


