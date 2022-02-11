import tensorflow as tf
import pandas as pd
import numpy as np
tf.set_random_seed(42)
from sklearn.model_selection import train_test_split

#1. 데이터
path = '../_data/kaggle/bike/'  
train = pd.read_csv(path+'train.csv')  # print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
x_data = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
y_data = train['count']
y_data = y_data.values.reshape(10886,1)
x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=104)


x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])


w1 = tf.compat.v1.Variable(tf.random.normal([8,64]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([64]), name = 'bias1')

Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([64,32]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([32]), name = 'bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([32,16]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([16]), name = 'bias3')

Hidden_layer3 = tf.nn.selu(tf.matmul(Hidden_layer2,w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([16,8]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([8]), name = 'bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3,w4) + b4

w5 = tf.compat.v1.Variable(tf.random.normal([8,1]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias5')

#2. 모델 구성
hypothesis = tf.matmul(Hidden_layer4, w5) + b5 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.AdamOptimizer(learning_rate= 1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(20001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w5, b5],
                                         feed_dict = {x : x_train,
                                                      y:y_train})
    if epochs % 2000== 0:
        print(epochs, loss_val,)

#4. 평가, 예측
y_pred = tf.matmul(x, w5) + b5
y_pred_data = sess.run(y_pred, feed_dict={x : x_test})
# print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_test , y_pred_data)
print('r2 :', r2)