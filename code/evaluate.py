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

total_cost = []
energy_cost = []
time_cost = []
done = False
for i in range(1000):
    env.reset()
    while not done:
        action = random.randrange(1, 10)
        state, reward, done, truncated, info = env.step(action)
    time_cost.append(info["total_delay"])
    energy_cost.append(info["energy_cost"])
    total_cost.append(info["energy_cost"] + info["total_delay"])

print(
    f"total_cost:{sum(total_cost) / len(total_cost)} energy_cost:{sum(energy_cost) / len(energy_cost)} time_cost:{sum(time_cost) / len(time_cost)}")
