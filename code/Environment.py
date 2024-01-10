import math
import random
import sys

import gymnasium
from gymnasium import spaces

from Node import *
from PathCreator import PathCreator


class OffloadingEnv(gymnasium.Env):
    def __init__(self, config, is_print=False):
        super(OffloadingEnv, self).__init__()
        # 加载配置文件
        self.config = config
        # 设置随机种子
        random.seed(self.config['random_seed'])
        # 定义各个维度的取值范围
        self.dim1_range = self.config['task_dimensions']
        self.dim2_range = self.config['max_loss'] - self.config['min_loss'] + 1
        self.dim3_values = len(self.config['vehicle_config']['C_n']) + 1
        self.dim4_values = len(self.config['vehicle_config']['P_n']) + 1
        self.dim5_values = len(self.config['vehicle_config']['P_n']) + 1
        self.dim6_values = len(self.config['vehicle_config']['C_n']) + 1
        self.dim7_values = len(self.config['vehicle_config']['P_n']) + 1
        self.dim8_values = len(self.config['vehicle_config']['E_n']) + 1

        # 定义状态空间为 MultiDiscrete
        self.observation_space = spaces.MultiDiscrete([
            self.dim1_range,
            self.dim2_range,
            self.dim3_values,
            self.dim4_values,
            self.dim5_values,
            self.dim6_values,
            self.dim7_values,
            self.dim8_values
        ])

        self.action_space = spaces.Discrete(len(self.config['task_split_granularity']))
        self.current_step = 1
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
        for i in self.vehicle_paths:
            for j in range(len(i)):
                i[j] = np.array([i[j][0] * 10, i[j][1] * 10, i[j][2] * 10])

        # 时间线 随机一个时间点用于表示一个不确定的时间点进入任务计算状态
        self.time_line = random.randint(0, 1500)
        self.total_delay_of_task = 0
        self.total_energy_cost_of_task = 0
        self.total_reward_of_episode = 0
        self.max_delay = 1
        self.max_cost = 1
        # 节点定义：4个无人机，20个车辆
        self.nodes = []
        # 定义无人机
        for i in range(len(self.config['uav_config']['pos'])):
            self.nodes.append(UAV(self.config, i))
        # 定义车
        for i in range(self.config['vehicle_path_config']["vehicle_num"]):
            self.nodes.append(
                Vehicle(self.config, i + len(self.config['uav_config']['pos']), self.vehicle_paths[i], self.time_line,
                        len(self.config['uav_config']['pos'])))

        # 定义任务
        self.data_size = self.config['data_size']

        # 获得一些最大最小值以便于状态归一化
        self.max_loss = self.config['max_loss']
        self.min_loss = self.config['min_loss']
        self.max_c_n = max(self.config['vehicle_config']['C_n'])
        self.max_c_n = max(self.max_c_n, self.config['uav_config']['C_n'])

        self.min_c_n = min(self.config['vehicle_config']['C_n'])
        self.min_c_n = min(self.min_c_n, self.config['uav_config']['C_n'])

        self.max_p_n = max(self.config['vehicle_config']['P_n'])
        self.max_p_n = max(self.max_p_n, self.config['uav_config']['P_n'])

        self.min_p_n = min(self.config['vehicle_config']['P_n'])
        self.min_p_n = min(self.min_p_n, self.config['uav_config']['P_n'])

        self.max_e_n = max(self.config['vehicle_config']['E_n'])
        self.max_e_n = max(self.max_e_n, self.config['uav_config']['E_n'])

        self.min_e_n = min(self.config['vehicle_config']['E_n'])
        self.min_e_n = min(self.min_e_n, self.config['uav_config']['E_n'])
        self.task_interval = self.config['data_size'] / self.config['task_dimensions']

        self.e_mean = 0.0
        self.e_std = 1.0
        self.t_mean = 0.0
        self.t_std = 1.0
        self.epsilon = 1e-8
        self.global_step = 1
        self.is_print = is_print

    def step(self, action):

        # 处理数据值
        data_size_on_local, data_size_on_remote = self.deal_data_size(action)

        # 检查是否为结束状态
        done = bool(data_size_on_remote <= 0)  # 类型为<class 'numpy.bool_'>，所以需要转一下
        # 计算奖励值
        reward, energy, time = self.get_reward(self.current_node, self.target_node, data_size_on_local,
                                               data_size_on_remote, done)
        # 更新车辆的位置与时间线
        time_step = math.ceil(time / self.config['vehicle_path_config']['time_slot'])
        self.time_line += time_step
        # 判断是否超出范围
        assert self.time_line < len(self.vehicle_paths[0])
        for i in self.nodes:
            if i.type == 'vehicle':
                i.run_step(self.time_line)
        self.total_energy_cost_of_task += energy
        self.total_delay_of_task += time
        self.total_reward_of_episode += reward

        # 环境进入下一个状态
        self.current_node = self.target_node
        self.target_node = self.choose_target_node(self.current_node)

        # 节点切换以后对应数据大小也切换
        self.data_size = data_size_on_remote

        # 构造下一个状态
        state = self.construct_state(self.current_node, self.target_node, self.data_size)
        truncated = False  # 是否因为最大步数限制被提前终止
        info = {}  # 附加信息字典
        # 打印日志信息
        em = '\n' if self.current_step % 10 == 0 or done else ''
        if done:
            info["total_delay"] = self.total_delay_of_task
            info["energy_cost"] = self.total_energy_cost_of_task
            info["done"] = done
            info["episode_reward"] = self.total_reward_of_episode
            info["episode_length"] = self.current_step
            if self.is_print:
                # print("finished  action:" + str(action) + " " + str(
                #     self.current_node.type + " " + str(
                #         self.current_node.id)) + "(target:" + self.target_node.type + str(
                #     self.target_node.id) + ")")

                print(em+"\033[92m timeline:" + str(self.time_line) + " total delay: " + str(
                    self.total_delay_of_task) + " energy cost:" + str(
                    self.total_energy_cost_of_task) + " episode reward:" + str(
                    self.total_reward_of_episode) + "\033[0m")
        else:
            # 更新step
            self.current_step += 1
            self.global_step += 1
            if self.is_print:
                print("-->"+str(self.current_node.type + " " + str(self.current_node.id)).center(11), end=em)

        return state, reward, done, truncated, info

    def reset(self, seed=1):
        # 重置环境状态
        self.time_line = random.randint(0, 1500)
        self.total_delay_of_task = 0
        self.total_energy_cost_of_task = 0
        self.total_reward_of_episode = 0
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
        if self.is_print:
            print("\033[93m" + "-" * 50 + "\033[0m")
            print("\033[93m" + "|" + "episode".center(20) + "|" + str(
                str(self.episode) + "(" + str(self.global_step) + ")").center(27) + "|" + "\033[0m")
            print("\033[93m" + "-" * 50 + "\033[0m")
            # 打印卸载路线
            print(str(self.current_node.type + " " + str(self.current_node.id)).center(11), end='')
        return initial_state, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

    def choose_target_node(self, current_node):
        min_value = sys.maxsize
        min_index = -1
        # print("********************************one choose*****************************")
        vehicle_weight = {}
        for i in range(len(self.nodes)):
            if current_node.id == self.nodes[i].id or current_node.node_is_in_range(self.nodes[i]) is False:
                continue
            assert current_node.id != self.nodes[i].id
            # print(f" \n{self.current_node.type}{self.current_node.id}***********", self.nodes[i].type, self.nodes[i].id,
            #         end=" ")
            e = self.nodes[i].energy_consumption_of_node_computation(
                1) + current_node.energy_consumption_of_node_transmission(1, self.nodes[i])
            t = current_node.target_node_offloading_time(1, 1, self.nodes[i])
            # print(" e:", e, " t:", t, end=" ")
            weight = self.config['e_weight'] * e + self.config['t_weight'] * t
            # print("part 1:"+str(self.config['e_weight'] * e)+" part 2:"+str(self.config['t_weight'] * t)+" weight:"+str(weight))
            if self.nodes[i].type == "vehicle":
                vehicle_weight[i] = weight
            # 这里是必须的，因为无法保证没有车辆的情况
            if weight < min_value:
                min_value = weight
                min_index = i
        # # 如果此时当前无人机拥有车辆，那么应该优先选择车辆
        # if len(vehicle_weight) != 0:
        #     min_value = sys.maxsize
        #     min_index = -1
        #     for i, weight in vehicle_weight.items():
        #         if weight < min_value:
        #             min_value = weight
        #             min_index = i
        # print("result:",self.nodes[min_index].type+str(self.nodes[min_index].id))
        assert self.nodes[min_index].id != current_node.id
        return self.nodes[min_index]

    def construct_state(self, current_node, target_node, data_size):
        s_t = current_node.w * data_size
        max_w = self.config['uav_config']['w'] if self.config['uav_config']['w'] > self.config['vehicle_config'][
            'w'] else self.config['vehicle_config']['w']

        s_t = int(s_t / (self.task_interval * max_w))
        s_t = s_t / self.config['task_dimensions']

        loss = math.ceil(current_node.get_path_loss(target_node))
        # print("currentpos:",current_node.position," targetpos:",target_node.position," type:",current_node.type,target_node.type," loss:",loss," dis:",current_node.get_dis(current_node.position,target_node.position))
        assert loss in range(self.config['min_loss'], self.config['max_loss'] + 1), f"loss:{loss} 值超出范围了"
        loss = (loss - self.min_loss) / (self.max_loss - self.min_loss)

        c_nt = (current_node.C_n - self.min_c_n) / (self.max_c_n - self.min_c_n)
        p_nt = (current_node.P_n - self.min_p_n) / (self.max_p_n - self.min_p_n)
        e_nt = (current_node.E_n - self.min_e_n) / (self.max_e_n - self.min_e_n)

        c_nt_next = (target_node.C_n - self.min_c_n) / (self.max_c_n - self.min_c_n)
        p_nt_next = (target_node.P_n - self.min_p_n) / (self.max_p_n - self.min_p_n)
        e_nt_next = (target_node.E_n - self.min_e_n) / (self.max_e_n - self.min_e_n)
        # print(
        #     f"maxc:{self.max_c_n} minc:{self.min_c_n} currentc:{current_node.C_n} maxp:{self.max_p_n} minp:{self.min_p_n} currentp:{current_node.P_n} maxe:{self.max_e_n} mine:{self.min_e_n} currente:{current_node.E_n} targetc:{target_node.C_n} targetp:{target_node.P_n} targete:{target_node.E_n}"
        # )
        state = np.array([s_t, loss, c_nt, p_nt, e_nt, c_nt_next, p_nt_next, e_nt_next], dtype=np.float32)
        return state

    def get_reward(self, current_node, target_node, data_size_on_local, data_size_on_remote, done):
        assert current_node.id != target_node.id
        e1 = current_node.energy_consumption_of_node_computation(data_size_on_local)
        e2 = current_node.energy_consumption_of_node_transmission(data_size_on_remote, target_node)
        e = e1 + e2
        t = current_node.offloading_time(data_size_on_local, data_size_on_remote, target_node)
        time = t * self.config['t_weight']
        energy = e * self.config['e_weight']
        # 将值归一化
        # e_normalized, t_normalized = self.normalize_values(e, t)

        reward = - (e * self.config['e_weight'] + t * self.config['t_weight'])

        # print("e:", e, " t:",t ," e_mean:" + str(self.e_mean) + " e_std:" + str(self.e_std) + " t_mean:" + str(
        #     self.t_mean) + " t_std:" + str(self.t_std), " reward:", reward)
        return reward, energy, time

    def deal_data_size(self, action):
        task_split_granularity = self.config['task_split_granularity'][action]
        data_size_on_local = math.ceil(task_split_granularity * self.data_size)
        data_size_on_remote = self.data_size - data_size_on_local
        return data_size_on_local, data_size_on_remote

    def normalize_values(self, e, t):
        # 更新均值和标准差
        self.e_mean = (self.e_mean * self.global_step + e) / (self.global_step + 1)
        self.e_std = np.sqrt(
            ((self.e_std ** 2) * self.global_step + (e - self.e_mean) ** 2) / (self.global_step + 1))

        self.t_mean = (self.t_mean * self.global_step + t) / (self.global_step + 1)
        self.t_std = np.sqrt(
            ((self.t_std ** 2) * self.global_step + (t - self.t_mean) ** 2) / (self.global_step + 1))

        # 对 e 和 t 进行归一化
        normalized_e = (e - self.e_mean) / (self.e_std + self.epsilon)
        normalized_t = (t - self.t_mean) / (self.t_std + self.epsilon)
        return normalized_e, normalized_t

# todo reset的seed报错未解决
