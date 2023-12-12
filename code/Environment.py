import random
import gymnasium
from gymnasium import spaces
from PathCreator import PathCreator
import sys
from Node import *


class OffloadingEnv(gymnasium.Env):
    def __init__(self, config):
        super(OffloadingEnv, self).__init__()
        # 加载配置文件
        self.config = config
        # 设置随机种子
        random.seed(self.config['random_seed'])
        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=np.array([0, -np.inf, 0, 0, 0, 0, 0, 0]),
                                            high=np.array(
                                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(10)  # 动作空间大小为10
        self.current_step = 0
        self.episode = 0
        self.current_node = None
        self.target_node = None
        # 定义车辆的路径
        self.path_creator = PathCreator(self.config['vehicle_path_config']['car_speed'],
                                        self.config['vehicle_path_config']['run_time'],
                                        self.config['vehicle_path_config']['time_slot'],
                                        self.config['vehicle_path_config']['path_num'],
                                        self.config['vehicle_path_config']['forward_probability'],
                                        self.config['random_seed'])
        self.path_creator.createPath()
        self.vehicle_paths = self.path_creator.pathPoint

        # 时间线 随机一个时间点用于表示一个不确定的时间点进入任务计算状态
        self.time_line = random.randint(0, 1500)
        # 节点定义：4个无人机，20个车辆
        self.nodes = []
        # 定义无人机
        for i in range(len(self.config['uav_config']['pos'])):
            self.nodes.append(UAV(self.config, i))
        # 定义车
        for i in range(self.config['vehicle_path_config']["vehicle_num"]):
            self.nodes.append(Vehicle(self.config, i, self.vehicle_paths[i], self.time_line))
        # 定义任务
        self.data_size = self.config['data_size']

    def step(self, action):
        # print("执行了step：", self.current_step + 1, "次")
        task_split_granularity = self.config['task_split_granularity'][action]
        data_size_on_local = int(task_split_granularity * self.data_size)
        data_size_on_remote = self.data_size - data_size_on_local

        # 更新车辆的位置与时间线
        time = self.current_node.offloading_time(data_size_on_local, data_size_on_remote, self.target_node)
        time_step = int(time / self.config['vehicle_path_config']['time_slot'])
        self.time_line += time_step
        assert self.time_line < len(self.vehicle_paths[0])  # 保证不会超出轨迹点
        for i in self.nodes:
            if i.type == 'vehicle':
                i.run_step(self.time_line)

        time_step = 0

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
        done = bool(self.data_size == 0)  # 类型为<class 'numpy.bool_'>，所以需要转一下
        self.current_step += 1
        truncated = False  # 是否因为最大步数限制被提前终止
        info = {"reward": reward}  # 附加信息字典

        # 打印日志信息
        em = '\n' if self.current_step % 10 == 0 else ''
        print(str(self.current_node.type + " " + str(self.current_node.id)).ljust(11) + "-->", end=em)
        # print(str(self.current_node.type + " " + str(self.current_node.id)).ljust(11) +"size:"+str(self.data_size)+ "-->", end=em)
        if done:
            print("finished")
        return state, reward, done, truncated, info

    def reset(self, seed=1):
        # 重置环境状态
        self.time_line = random.randint(0, 1500)
        self.data_size = self.config['data_size']
        self.current_step = 1
        self.episode += 1
        # 重置车辆的位置
        for i in self.nodes:
            if i.type == 'vehicle':
                i.reset(self.time_line)
        # 随机选择一个无人机节点以接收一个任务
        node_index = random.randint(0, len(self.config['uav_config']['pos']) - 1)
        self.current_node = self.nodes[node_index]
        self.target_node = self.choose_target_node(self.current_node)

        initial_state = self.construct_state(self.current_node, self.target_node, self.data_size)  # 初始化状态
        info = {}
        # 打印日志信息
        print()
        print("\033[93m" + "-" * 50 + "\033[0m")
        print("\033[93m" + "|" + "episode".center(20) + "|" + str(self.episode).center(27) + "|" + "\033[0m")
        print("\033[93m" + "-" * 50 + "\033[0m")
        # print("\033[93m timeline:" + str(self.time_line) + "\033[0m")
        print("offloading_route")
        print(str(self.current_node.type + " " + str(self.current_node.id)).ljust(11) + "-->",
              end='')
        # print(str(self.current_node.type + " " + str(self.current_node.id)).ljust(11)+"size:"+str(self.data_size) + " -->",
        #       )
        return initial_state, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def choose_target_node(self, current_node):
        min_value = sys.maxsize
        min_index = -1
        print("\n choose:" + str(self.current_node.type + str(self.current_node.id)))
        for i in range(len(self.nodes)):
            if current_node.id == self.nodes[i].id or current_node.node_is_in_range(self.nodes[i]) is False:
                continue
            assert current_node.id != self.nodes[i].id
            e = self.nodes[i].energy_consumption_of_node_computation(
                1) + current_node.energy_consumption_of_node_transmission(1, self.nodes[i])
            t = current_node.offloading_time(1, 1, self.nodes[i])
            weight = self.config['node_choose_config']['e_weight'] * e + self.config['node_choose_config'][
                't_weight'] * t
            if self.nodes[i].type == 'vehicle':
                print("time_line:", self.time_line, self.nodes[i].type, " ", self.nodes[i].id, " ",
                      self.nodes[i].position, end='')
            if weight < min_value:
                min_value = weight
                min_index = i
        print()
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
        assert current_node.id != target_node.id
        e = current_node.energy_consumption_of_node_computation(
            data_size_on_local) + current_node.energy_consumption_of_node_transmission(data_size_on_remote, target_node)
        t = current_node.offloading_time(data_size_on_local, data_size_on_remote, target_node)
        return e * self.config['reward_config']['e_weight'] + t * self.config['reward_config']['t_weight']
#
#
# myenv = OffloadingEnv()
# check_env(myenv)
