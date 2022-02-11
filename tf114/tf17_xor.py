import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2,1]), name = 'weight')   
b = tf.compat.v1.Variable(tf.zeros([1]), )
# bias = 행렬의 덧셈
# 입력 값은 placeholder
# zeros는 통상적으로 bias에 넣는다. 
#2. 모델 구성


