import os
import random
from evaluate import Evaluate, Tools
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from Environment import OffloadingEnv
from LoadParameters import *
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


# TODO　当奖励归一化以后，但是目标选取策略似乎还没有改变
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    t_weight: int = 1
    e_weight: int = 1
    is_comparison_experiment: bool = False

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def make_env(is_print, seed):
    env = OffloadingEnv(config, is_print)
    env.action_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    loss_set = []
    reward_set = []
    time_set = []
    energy_set = []
    total_set = []
    length_set = []

    config = load_parameters()
    args = tyro.cli(Args)
    # 设置t的权重
    config["t_weight"] = args.t_weight * 0.5+5
    config["e_weight"] = 1
    config["is_comparison_experiment"] = args.is_comparison_experiment
    log_path = f"../log/e_weight_{config['e_weight']}_t_weight_{config['t_weight']}"
    os.makedirs(log_path, exist_ok=True)
    model_path = "../model/"
    os.makedirs(model_path, exist_ok=True)
    print("t_weight:", config["t_weight"], " e_weight:", config["e_weight"])

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    training_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())  # 用于设置log名称
    run_name = f"{training_time}"

    writer = SummaryWriter(f"../runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(False, config['random_seed'])
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if random.random() < epsilon:
            actions = np.array([random.randint(0, 10) for _ in range(args.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=0).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, done, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if done:
            writer.add_scalar("reward", infos["episode_reward"], global_step)
            writer.add_scalar("episodic_length", infos["episode_length"], global_step)
            writer.add_scalar("costs/time_cost", infos["total_delay"], global_step)
            writer.add_scalar("costs/energy_cost", infos["energy_cost"], global_step)
            writer.add_scalar("costs/total_cost", infos["energy_cost"] + infos["total_delay"], global_step)
            reward_set.append(infos["episode_reward"])
            time_set.append(infos["total_delay"])
            energy_set.append(infos["energy_cost"])
            total_set.append(infos["energy_cost"] + infos["total_delay"])
            length_set.append(infos["episode_length"])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        rb.add(obs, real_next_obs, actions, rewards, done, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("training/loss", loss, global_step)
                    writer.add_scalar("training/q_values", old_val.mean().item(), global_step)
                    loss_set.append(loss)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
        if done:
            obs, _ = envs.reset(seed=args.seed)  # 如果本episode结束则重置
        if (global_step + 1) >= 400000 and (global_step + 1) % 100000 == 0:
            torch.save(q_network.state_dict(),
                       model_path + f"step-{global_step + 1}-tweight-{config['t_weight']}-eweight-{config['e_weight']}")
            # 开始验证并记录验证的数据
            evaluator = Evaluate()
            evaluator.do_evaluate(
                model_path + f"step-{global_step + 1}-tweight-{config['t_weight']}-eweight-{config['e_weight']}",
                log_path, QNetwork)

    envs.close()
    writer.close()

    # 绘制训练曲线
    N = 100  # 数据采样频率
    tools = Tools()
    tools.draw_training_figs(loss_set, 1, "loss", "step", "loss", log_path)
    tools.draw_training_figs(total_set, N, "total_cost", "step", "total_cost", log_path)
    tools.draw_training_figs(time_set, N, "time_cost", "step", "time_cost", log_path)
    tools.draw_training_figs(energy_set, N, "energy_cost", "step", "energy_cost", log_path)
    tools.draw_training_figs(reward_set, N, "reward", "step", "reward", log_path)
    tools.draw_training_figs(length_set, N, "episode_length", "step", "episode_length", log_path)
