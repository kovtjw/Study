from tkinter import Variable
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.utils import to_categorical # 1부터 시작된다. 
# one hot은 0부터
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000,28, 28, 1).astype('float32')/255.

x = tf.compat.v1.placeholder(tf.float32,[None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32,[None, 10])

#2. 모델 구성

w1 = tf.get_variable('w1', shape = [2,2,1,32])
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1], padding = 'SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

print(w1)  # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)  # Tensor("Conv2D:0", shape=(?, 28, 28, 64), dtype=float32)
print(L1_maxpool)  # Tensor("MaxPool:0", shape=(?, 14, 14, ), dtype=float32)

# Layer 2
w2 = tf.get_variable('w2', shape = [3,3,32,16])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L2_maxpool)  # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# Layer 3
w3 = tf.get_variable('w3', shape = [3,3,16,32])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding = 'SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(L3_maxpool)   # Tensor("MaxPool2d_2:0", shape=(?, 4, 4, 32), dtype=float32)

# # Layer 4
# w4 = tf.get_variable('w4', shape = [3,3,32,32])
# L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding = 'SAME')
# L4 = tf.nn.elu(L4)
# L4_maxpool = tf.nn.max_pool2d(L4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
# print(L4_maxpool)   # Tensor("MaxPool2d_3:0", shape=(?, 2, 2, 32), dtype=float32)

# Flatten 
L_flat = tf.reshape(L3_maxpool, [-1, 4*4*32])
print('Flatten :', L_flat)   # Flatten : Tensor("Reshape:0", shape=(?, 128), dtype=float32)

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape = [4*4*32, 64])
b5 = tf.Variable(tf.random_normal([64], name='b5'))

L5 = tf.matmul(L_flat,w5)+ b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob = 0.7)

# Layer6 DNN
w6 = tf.compat.v1.get_variable('w6', shape = [64, 32])

b6 = tf.Variable(tf.random_normal([32], name='b6'))
L6 = tf.matmul(L5,w6)+ b6
L6 = tf.nn.relu(L6)
L6 = tf.nn.dropout(L6, keep_prob = 0.7)

# Layer7 softmax
w7 = tf.compat.v1.get_variable('w7', shape = [32, 10])
b7 = tf.Variable(tf.random_normal([10], name='b7'))
L7 = tf.matmul(L6,w7)+ b7
hypothesis = tf.nn.softmax(L7)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))  # categorical_crossentropy

optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) 

training_epochs =100
batch_size = 100
total_batch = int(len(x_train)/batch_size)

#3-2. 훈련
for epoch in range(training_epochs):
    avg_loss = 0
    
    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size    # 0
        end = start + batch_size  # 100
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        
        feed_dict = {x:batch_x, y:batch_y}   # 단순 변수
        
        batch_loss, _ = sess.run([loss, optimizer], feed_dict = feed_dict)
        
        avg_loss += batch_loss / total_batch    # += : 더한 것을 넣어준다.
        
    print('Epoch :', '%04d' %(epoch + 1), ';loss : {:.9f}'.format(avg_loss))
print('훈련 끗!')

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))


