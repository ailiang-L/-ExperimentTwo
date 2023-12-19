from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from Environment import OffloadingEnv
import random
import torch
from LoadParameters import *
from CallBack import *
import time

# 加载配置文件
config = load_parameters()
# 设置整体随机种子
seed_value = config['random_seed']
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

# 创建环境
env = OffloadingEnv(config)

# 创建向量化环境
vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed_value)

# 创建DQN模型，并设置种子
model = DQN("MlpPolicy", vec_env, verbose=0, seed=seed_value, tensorboard_log='../log/', learning_rate=2.5e-4,
            buffer_size=30000, gamma=0.99, tau=1.0, batch_size=128, learning_starts=20000, exploration_initial_eps=1,
            exploration_final_eps=0.05, exploration_fraction=0.5, train_freq=10, target_update_interval=1500)

# 定义回调类
call_back = CustomCallback(0)
# 设置log名
training_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
# 训练模型
model.learn(total_timesteps=5000000, callback=call_back, tb_log_name=training_time, log_interval=4)
