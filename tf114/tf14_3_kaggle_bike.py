from sklearn.datasets import load_diabetes
import tensorflow as tf
import pandas as pd
import numpy as np
tf.set_random_seed(42)

#1. 데이터
path = '../_data/kaggle/bike/'  
train = pd.read_csv(path+'train.csv')  # print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
x_data = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
y_data = train['count']
y_data = y_data.values.reshape(10886,1)

print(x_data.shape, y_data.shape)  # (10886, 8) (10886,)

x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([8,1]), name = 'weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델 구성
hypothesis = tf.matmul(x, w) + b 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(50001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                         feed_dict = {x : x_data,
                                                      y:y_data})
    if epochs % 100 == 0:
        print(epochs, loss_val,w_val, b_val)
        

#4. 평가, 예측
y_pred = tf.matmul(x, w) + b
y_pred_data = sess.run(y_pred, feed_dict={x : x_data})
print('y_pred_data :',y_pred_data)


from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data , y_pred_data)
print('r2 :', r2)

sess.close()
