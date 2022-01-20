import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
(x_train, _),(x_test, _) = mnist.load_data() # 오류가 나지 않게 언더바처리함 >> y는 안가져옴
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0)
print(x.shape)  # (70000, 28, 28)
x = x.reshape(70000,784)
print(x.shape)
pca = PCA(n_components=784)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
print(sum(pca_EVR))
print(pca_EVR)
cumsum = np.cumsum(pca_EVR)
print(cumsum)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# # plt.plot(pca_EVR)
# plt.grid()
# plt.show()
print(np.argmax(cumsum))  # 712 
print(np.argmax(cumsum>0.999)) # 330
# print(np.argwhere(cumsum))

#########################################################
## 실습                                                 #
## pca를 통해 0.95 이상인 n_components 가 몇 개인지 찾기#
# 0.95 >> 153                                           #
# 0.99 >> 330                                           #
# 0.999 >> 485                                          #
# 1.0                                                   #
#########################################################