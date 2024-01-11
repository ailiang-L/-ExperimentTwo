import os

from Environment import OffloadingEnv
from LoadParameters import *
import matplotlib.pyplot as plt
import pandas as pd
import random

import numpy as np
import torch


class Evaluate:
    def do_evaluate(self, model_path="../model/step-900000-tweight-20.0-eweight-1.0", log_path='../output',
                    QNetwork=None):
        log_path = log_path + "/comparison/"
        os.makedirs(log_path, exist_ok=True)
        # 加载配置文件
        config = load_parameters()
        # 通过 '-' 进行分割，得到一个包含各部分的列表
        parts = model_path.split('-')
        # 找到包含 'tweight' 和 'eweight' 的部分
        t_weight_index = parts.index('tweight')
        e_weight_index = parts.index('eweight')

        config["t_weight"] = eval(parts[t_weight_index + 1])  # 获取 'tweight' 后面的值
        config["e_weight"] = eval(parts[e_weight_index + 1])  # 获取 'eweight' 后面的值
        step = eval(parts[t_weight_index - 1])

        # 创建环境
        envs = [OffloadingEnv(config) for i in range(2)]

        # DQN 策略
        self.set_seed(config)
        if QNetwork is None:
            from CleanRL import QNetwork
        reward, length, time1, energy1, total1 = self.DQN_evaluate(model_path, envs[0], eval_episodes=100,
                                                                   Model=QNetwork)

        # 随机策略
        self.set_seed(config)
        time2, energy2, total2 = self.RM_evaluate(envs[1])

        plt.switch_backend('agg')  # 将绘图引擎切换为无头模式
        # 绘制time1和time2的对比图并保存
        plt.figure(figsize=(8, 6))
        plt.plot(time1, label='DQN')
        plt.plot(time2, label='RM')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Comparison of time cost')
        plt.legend()
        plt.grid(True)
        plt.savefig(log_path + f'{step}_time_comparison.png')  # 保存图片为 time_comparison.png 文件
        plt.close()  # 关闭图形，防止显示在屏幕上

        # 绘制energy1和energy2的对比图并保存
        plt.figure(figsize=(8, 6))
        plt.plot(energy1, label='DQN')
        plt.plot(energy2, label='RM')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Comparison of energy cost')
        plt.legend()
        plt.grid(True)
        plt.savefig(log_path + f'{step}_energy_comparison.png')  # 保存图片为 energy_comparison.png 文件
        plt.close()  # 关闭图形，防止显示在屏幕上

        # 绘制total1和total2的对比图并保存
        plt.figure(figsize=(8, 6))
        plt.plot(total1, label='DQN')
        plt.plot(total2, label='RM')
        plt.xlabel('episode')
        plt.ylabel('Values')
        plt.title('Comparison of total cost')
        plt.legend()
        plt.grid(True)
        plt.savefig(log_path + f'{step}_total_comparison.png')  # 保存图片为 total_comparison.png 文件
        plt.close()  # 关闭图形，防止显示在屏幕上

    def RM_evaluate(self, env):
        total_cost = []
        energy_cost = []
        time_cost = []
        for i in range(100):
            env.reset()
            done = False
            while not done:
                action = random.randint(0, 10)
                state, reward, done, truncated, info = env.step(action)

            # print(info["energy_cost"],info["total_delay"])
            time_cost.append(info["total_delay"])
            energy_cost.append(info["energy_cost"])
            total_cost.append(info["energy_cost"] + info["total_delay"])
        # print("total_avg_cost:", sum(total_cost) / len(total_cost), "time_avg_cost:", sum(time_cost) / len(time_cost),
        #       "energy_avg_cost:", sum(energy_cost) / len(energy_cost))
        return time_cost, energy_cost, total_cost

    def DQN_evaluate(self,
                     model_path: str,
                     envs: OffloadingEnv,
                     eval_episodes: int,
                     Model: torch.nn.Module,
                     device: torch.device = torch.device("cpu"),
                     epsilon: float = 0.05,
                     ):
        model = Model(envs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        obs, _ = envs.reset()
        episodic_reward = []
        episode_length = []
        time_cost = []
        energy_cost = []
        total_cost = []
        while len(episodic_reward) < eval_episodes:
            if random.random() < epsilon:
                actions = np.array([envs.action_space.sample() for _ in range(1)])
            else:
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=0).cpu().numpy()
            next_obs, _, done, _, infos = envs.step(actions)
            obs = next_obs

            if done:
                reward = infos["episode_reward"]
                episodic_length = infos["episode_length"]
                time = infos["total_delay"]
                energy = infos["energy_cost"]
                total = infos["energy_cost"] + infos["total_delay"]
                # 记录
                episodic_reward.append(reward)
                episode_length.append(episodic_length)
                time_cost.append(time)
                energy_cost.append(energy)
                total_cost.append(total)
                obs, _ = envs.reset()
        return episodic_reward, episodic_length, time_cost, energy_cost, total_cost

    def set_seed(self, config):
        # 设置整体随机种子
        seed_value = config['random_seed']
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)


class Tools:
    def __init__(self):
        plt.switch_backend('agg')  # 将绘图引擎切换为无头模式

    def draw_training_figs(self, data_set, sample_frequency, title, x_label, y_label, save_path):
        plt.figure(figsize=(18, 16))
        # Convert GPU tensors to NumPy arrays by moving them to CPU
        if isinstance(data_set[0], torch.Tensor):
            set_cpu = [element.cpu().detach().numpy() for element in data_set]
        else:
            set_cpu = data_set[::sample_frequency]
        x = [i * sample_frequency for i in range(len(set_cpu))]
        plt.plot(x, set_cpu, label=title)

        # 计算平滑曲线
        window_size = 100  # 可根据实际情况调整窗口大小
        smoothed_data = np.convolve(set_cpu, np.ones(window_size) / window_size, mode='valid')
        x_smoothed = [i * sample_frequency + window_size / 2 for i in range(len(smoothed_data))]

        plt.plot(x_smoothed, smoothed_data, label=f'Smoothed {title}')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title + " of training")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path + f"/{title}_with_smoothing.png")
        plt.close()

# data_set = [random.randint(1, 60) for i in range(1000000)]
# tools = Tools()
# tools.draw_training_figs(data_set, 200, "test", "step", "test", "./")
