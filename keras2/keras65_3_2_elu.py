import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha):
    return (x>0)*x + (x<=0)*(alpha *(np.exp(x)-1))
# 음수가 나올 수 없는 것을 정의 함 >> 음수 지역의 값은 소멸되버림

x = np.arange(-5, 5, 0.1)
alpha = 1
y = elu(x, alpha)

plt.plot(x,y)
plt.grid()
plt.show()
