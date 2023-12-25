import numpy as np
import matplotlib.pyplot as plt

# 生成 x 值范围（从0.1到200，间隔0.1）
x = np.linspace(0.1, 200, 1000)
# 计算对数函数的值
y = np.log(x)

# 绘制对数函数的图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = log(x)')
plt.xlabel('x')
plt.ylabel('log(x)')
plt.title('Plot of log(x) for x in range (0.1 to 200)')
plt.grid(True)
plt.legend()
plt.show()
