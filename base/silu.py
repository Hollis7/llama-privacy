import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.stats import norm


def gelu(x):
    return x * norm.cdf(x)


def relu(x):
    return np.maximum(0, x)


def swish(x, beta=1):
    return x * (1 / (1 + np.exp(-beta * x)))


def swiglu(x, W, V, b, c):
    return swish(x * W + b) * (x * V + c)

x_values = np.linspace(-5, 5, 500)
relu_values = relu(x_values)
swish_values = swish(x_values)
swish_values2 = swish(x_values, beta=0.5)
swiglu_values = swiglu(x_values, 1, 1, 0,
                       0)  # Here you need to set the parameters W, V, b, and c according to your needs

plt.plot(x_values, relu_values, label='ReLU')
plt.plot(x_values, swish_values, label='Swish')
plt.plot(x_values, swish_values2, label='Swish (beta=0.5)')
plt.plot(x_values, swiglu_values, label='SwiGLU')
plt.title("ReLU, Swish, and SwiGLU Activation Functions")
plt.xlabel("x")
plt.ylabel("Activation")
plt.grid()
plt.legend()
# 保存图像为文件
# 保存图像
plt.savefig('./imgs/silu_plt.png')

# 显示图像但不阻塞
plt.show()