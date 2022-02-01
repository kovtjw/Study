import numpy as np

a = np.array(range(1,11))  # 1~10
size = 3  # 5개씩 자른다.

def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): # 1~10 - 5 +1 //len은 0부터 시작하기 때문에 +1을 해주어야 함
        subset = data[i : (i+size)]      # [1 : 6] = 1,2,3,4,5 
        aaa.append(subset)                  # 1,2,3,4,5에 []를 붙인다.
    return np.array(aaa)                    # numpy 배열로 리턴한다.

dataset = split_x(a, size)                  # 대입했을 때
print(dataset)

bbb = split_x(a, size)                  # 대입했을 때
print(bbb.shape)

x = bbb[:,:-1]  # :, = 0:-1, 모든 행부터 4까지 
y = bbb[:,-1]   

print(x.shape,y.shape) # (6, 4) (6,)