import os

from Environment import OffloadingEnv
from LoadParameters import *
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import numpy as np
import torch


class Evaluate:
    def do_evaluate(self, model_path="../model/step-799999-tweight-10.0-eweight-1.0-comparison-False",
                    log_path='../output', is_plotting=True, data_size=1e8):
        log_path = log_path + "/comparison/"
        os.makedirs(log_path, exist_ok=True)
        # 加载配置文件
        config = load_parameters()
        config["data_size"] = data_size
        # 通过 '-' 进行分割，得到一个包含各部分的列表
        parts = model_path.split('-')
        # 找到包含 'tweight' 和 'eweight' 的部分
        t_weight_index = parts.index('tweight')
        e_weight_index = parts.index('eweight')

        config["t_weight"] = eval(parts[t_weight_index + 1])  # 获取 'tweight' 后面的值
        config["e_weight"] = eval(parts[e_weight_index + 1])  # 获取 'eweight' 后面的值
        config["is_comparison_experiment"] = False
        step = eval(parts[t_weight_index - 1])

        # 创建环境
        envs = [OffloadingEnv(config) for i in range(3)]

        # DQN 策略
        self.set_seed(config)
        reward, length, time1, energy1, total1 = self.DQN_evaluate(model_path, envs[0], eval_episodes=100,
                                                                   Model=QNetwork)

        # 随机策略
        self.set_seed(config)
        time2, energy2, total2 = self.RM_evaluate(envs[1])

        # 只有无人机的对照实验
        self.set_seed(config)
        envs[2].is_comparison_experiment = True
        reward3, length3, time3, energy3, total3 = self.DQN_evaluate(model_path, envs[2], eval_episodes=100,
                                                                     Model=QNetwork)
        if is_plotting:
            plt.switch_backend('agg')  # 将绘图引擎切换为无头模式
            # 绘制time1和time2的对比图并保存
            plt.figure(figsize=(8, 6))
            plt.plot(time1, label='DQN')
            plt.plot(time2, label='RM')
            plt.plot(time3, label='WV')
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title('Comparison of time cost')
            plt.legend()
            plt.grid(True)
            plt.savefig(
                log_path + f'{step}_time_comparison-datasize-{config["data_size"]}.png')  # 保存图片为 time_comparison.png 文件
            plt.close()  # 关闭图形，防止显示在屏幕上

            # 绘制energy1和energy2的对比图并保存
            plt.figure(figsize=(8, 6))
            plt.plot(energy1, label='DQN')
            plt.plot(energy2, label='RM')
            plt.plot(energy3, label='WV')
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title('Comparison of energy cost')
            plt.legend()
            plt.grid(True)
            plt.savefig(
                log_path + f'{step}_energy_comparison-datasize-{config["data_size"]}.png')  # 保存图片为 energy_comparison.png 文件
            plt.close()  # 关闭图形，防止显示在屏幕上

            # 绘制total1和total2的对比图并保存
            plt.figure(figsize=(8, 6))
            plt.plot(total1, label='DQN')
            plt.plot(total2, label='RM')
            plt.plot(total3, label='WV')
            plt.xlabel('episode')
            plt.ylabel('Values')
            plt.title('Comparison of total cost')
            plt.legend()
            plt.grid(True)
            plt.savefig(
                log_path + f'{step}_total_comparison-datasize-{config["data_size"]}.png')  # 保存图片为 total_comparison.png 文件
            plt.close()  # 关闭图形，防止显示在屏幕上
        return np.mean(time1), np.mean(time2), np.mean(time3), np.mean(energy1), np.mean(energy2), np.mean(
            energy3), np.mean(total1), np.mean(total2), np.mean(total3)

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

    def DQN_evaluate_comparison(self,
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
        # 标定为对照实验
        envs.is_comparison_experiment = True
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


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        x = x.float()
        return self.network(x)


if __name__ == "__main__":
    data_size = [4e8, 4.5e8, 5e8, 5.5e8, 6e8]
    DQN_time_list = []
    DQN_energy_list = []
    DQN_total_list = []
    RM_time_list = []
    RM_energy_list = []
    RM_total_list = []
    WV_time_list = []
    WV_energy_list = []
    WV_total_list = []
    for i in range(len(data_size)):
        evaluator = Evaluate()
        time1, time2, time3, energy1, energy2, energy3, total1, total2, total3 = evaluator.do_evaluate(is_plotting=True,
                                                                                                       data_size=
                                                                                                       data_size[i])
        DQN_time_list.append(time1)
        DQN_energy_list.append(energy1)
        DQN_total_list.append(total1)

        RM_time_list.append(time2)
        RM_energy_list.append(energy2)
        RM_total_list.append(total2)

        WV_time_list.append(time3)
        WV_energy_list.append(energy3)
        WV_total_list.append(total3)

    plt.switch_backend('agg')  # 将绘图引擎切换为无头模式
    plt.figure(figsize=(8, 6))
    plt.plot(DQN_time_list, label='DQN')
    plt.plot(RM_time_list, label='RM')
    plt.plot(WV_time_list, label='WV')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Comparison of time cost')
    plt.legend()
    plt.grid(True)
    plt.savefig(
        '../output/' + f'time_comparison-datasize.png')  # 保存图片为 time_comparison.png 文件
    plt.close()  # 关闭图形，防止显示在屏幕上

    plt.switch_backend('agg')  # 将绘图引擎切换为无头模式
    plt.figure(figsize=(8, 6))
    plt.plot(DQN_energy_list, label='DQN')
    plt.plot(RM_energy_list, label='RM')
    plt.plot(WV_energy_list, label='WV')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Comparison of energy cost')
    plt.legend()
    plt.grid(True)
    plt.savefig(
        '../output/' + f'energy_comparison-datasize.png')  # 保存图片为 time_comparison.png 文件
    plt.close()  # 关闭图形，防止显示在屏幕上

    plt.switch_backend('agg')  # 将绘图引擎切换为无头模式
    plt.figure(figsize=(8, 6))
    plt.plot(DQN_total_list, label='DQN')
    plt.plot(RM_total_list, label='RM')
    plt.plot(WV_total_list, label='WV')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Comparison of total cost')
    plt.legend()
    plt.grid(True)
    plt.savefig(
        '../output/' + f'total_comparison-datasize.png')  # 保存图片为 time_comparison.png 文件
    plt.close()  # 关闭图形，防止显示在屏幕上
