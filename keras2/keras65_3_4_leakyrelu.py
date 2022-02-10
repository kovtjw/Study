import numpy as np
import matplotlib.pyplot as plt

def leakyrelu(x):
    return np.maximum(0.01 * x, x)
# 음수가 나올 수 없는 것을 정의 함 >> 음수 지역의 값은 소멸되버림

x = np.arange(-5, 5, 0.1)
y = leakyrelu(x)

plt.plot(x,y)
plt.grid()
plt.show()
