from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from Environment import OffloadingEnv
# 车辆轨迹数据
trajectory_list = [...]  # 这里应该是具体的轨迹数据

# 创建环境x
env = OffloadingEnv(trajectory_list)

# 创建向量化环境
vec_env = make_vec_env(lambda: env, n_envs=1)

# 创建DQN模型
model = DQN("MlpPolicy", vec_env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)
