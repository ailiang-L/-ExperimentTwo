import matplotlib.pyplot as plt
import numpy as np

# 生成服从均值为 0，标准差为 1.7 的正态分布随机变量数据
X_eta4 = np.random.normal(0, 1.7, 10000)  # 生成 10000 个样本

# 绘制直方图
plt.figure(figsize=(8, 6))
plt.hist(X_eta4, bins=50, density=True, alpha=0.7, color='skyblue')

# 绘制正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = (1 / (1.7 * np.sqrt(2 * np.pi))) * np.exp(-(x - 0) ** 2 / (2 * 1.7 ** 2))
plt.plot(x, p, 'k', linewidth=2)

plt.title('服从均值为 0，标准差为 1.7 的正态分布随机变量')
plt.xlabel('随机变量值')
plt.ylabel('概率密度')
plt.show()
