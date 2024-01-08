from Environment import OffloadingEnv
import random
import torch
from LoadParameters import *

# 加载配置文件
config = load_parameters()
# 设置整体随机种子
seed_value = config['random_seed']
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# 创建环境
env = OffloadingEnv(config)
config["t_weight"] = 8
total_cost = []
energy_cost = []
time_cost = []

for j in range(1):
    config["t_weight"] = 50
    for i in range(1000):
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

    print(
        f"total_cost:{sum(total_cost) / len(total_cost)} energy_cost:{sum(energy_cost) / len(energy_cost)} time_cost:{sum(time_cost) / len(time_cost)}")
