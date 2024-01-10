from typing import Callable

from CleanRL import QNetwork
from Environment import OffloadingEnv
from LoadParameters import *
import matplotlib.pyplot as plt
import pandas as pd
import random

import gymnasium as gym
import numpy as np
import torch


def evaluate1():
    e_avg = []
    t_avg = []
    sum_avg = []
    # 加载配置文件
    config = load_parameters()
    x = []
    x_e = []
    config["t_weight"] = 0
    config["e_weight"] = 1 - config["t_weight"]

    while config["t_weight"] <= 1:
        # 创建环境
        env = OffloadingEnv(config)
        # 设置整体随机种子
        seed_value = config['random_seed']
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        print("t_weight:", config["t_weight"])
        x.append(config["t_weight"])
        x_e.append(config["e_weight"])
        total_cost = []
        energy_cost = []
        time_cost = []
        for i in range(100):
            env.reset()
            # print("******************************start******************")
            done = False
            while not done:
                action = random.randint(0, 10)
                # print(env.current_node.type, env.current_node.id, "--", env.target_node.type, env.target_node.id)
                state, reward, done, truncated, info = env.step(action)

            # print(info["energy_cost"],info["total_delay"])
            time_cost.append(info["total_delay"])
            energy_cost.append(info["energy_cost"])
            total_cost.append(info["energy_cost"] + info["total_delay"])
        sum_avg.append(sum(total_cost) / len(total_cost))
        e_avg.append(sum(energy_cost) / len(energy_cost))
        t_avg.append(sum(time_cost) / len(time_cost))
        # 改变比重
        config["t_weight"] += 0.01
        config["e_weight"] = 1 - config["t_weight"]

    # 绘制曲线
    plt.plot(x, sum_avg, marker='o', label='total_cost')  # 绘制第一个集合的曲线
    plt.plot(x, e_avg, marker='o', label='energy_cost')  # 绘制第二个集合的曲线
    plt.plot(x, t_avg, marker='o', label='time_cost')  # 绘制第三个集合的曲线
    print(sum_avg)
    print(e_avg)
    print(t_avg)
    # 添加标题和标签
    plt.title('Three Sets Line Plot')  # 添加标题
    plt.xlabel('t_weight')  # 添加 X 轴标签
    plt.ylabel('cost')  # 添加 Y 轴标签

    # 添加图例
    plt.legend()  # 显示图例

    # 显示图形并保存为图片文件
    plt.savefig('../output/random_policy.png')  # 保存为 PNG 图片
    plt.show()
    # 生成DataFrame
    data = {'t_weight': x, 'e_weight': x_e, 'sum_avg': sum_avg, 'e_avg': e_avg, 't_avg': t_avg}
    df = pd.DataFrame(data)

    # 将DataFrame保存为CSV文件
    df.to_csv('../output/random_policy.csv', index=False)  # 保存为CSV文件


def evaluate2(env):
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


evaluate2(20, 1)


def evaluate(
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
            actions = np.array([envs.single_action_space.sample() for _ in range(1)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
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


def set_seed(config):
    # 设置整体随机种子
    seed_value = config['random_seed']
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


if __name__ == "__main__":
    # 加载配置文件
    config = load_parameters()
    # 创建环境
    envs = [OffloadingEnv(config) for i in range(2)]

    model_path = "../model/"

    # DQN 策略
    set_seed(config)
    reward, length, time1, energy1, total1 = evaluate(model_path, envs[0], eval_episodes=100, Model=QNetwork)

    # 随机策略
    set_seed(config)
    time2, energy2, total2 = evaluate2(envs[1])
    # 绘制time1和time2的对比图并保存
    plt.figure(figsize=(8, 6))
    plt.plot(time1, label='DQN')
    plt.plot(time2, label='RM')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Comparison of time1 and time2')
    plt.legend()
    plt.grid(True)
    plt.savefig('../output/time_comparison.png')  # 保存图片为 time_comparison.png 文件
    plt.close()  # 关闭图形，防止显示在屏幕上

    # 绘制energy1和energy2的对比图并保存
    plt.figure(figsize=(8, 6))
    plt.plot(energy1, label='DQN')
    plt.plot(energy2, label='RM')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Comparison of energy1 and energy2')
    plt.legend()
    plt.grid(True)
    plt.savefig('../output/energy_comparison.png')  # 保存图片为 energy_comparison.png 文件
    plt.close()  # 关闭图形，防止显示在屏幕上

    # 绘制total1和total2的对比图并保存
    plt.figure(figsize=(8, 6))
    plt.plot(total1, label='DQN')
    plt.plot(total2, label='RM')
    plt.xlabel('episode')
    plt.ylabel('Values')
    plt.title('Comparison of total1 and total2')
    plt.legend()
    plt.grid(True)
    plt.savefig('../output/total_comparison.png')  # 保存图片为 total_comparison.png 文件
    plt.close()  # 关闭图形，防止显示在屏幕上
