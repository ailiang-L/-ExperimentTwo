from Environment import OffloadingEnv
import random
import torch
from LoadParameters import *
import matplotlib.pyplot as plt
import pandas as pd

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
