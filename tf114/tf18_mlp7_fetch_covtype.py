import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
tf.set_random_seed(104)

#1. 데이터
datasets = fetch_covtype()
x_data = datasets.data
y_data = datasets.target.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

print(x_data.shape,y_data.shape) # (581012, 54) (581012, 7)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=104)

#2. 모델 구성
x = tf.placeholder('float', shape=[None, 54])
y = tf.placeholder('float', shape=[None, 7])

w1 = tf.compat.v1.Variable(tf.random.normal([54,128]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([128]), name = 'bias1')

Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([128,64]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.zeros([64]), name = 'bias2')

Hidden_layer2 = tf.nn.selu(tf.matmul(Hidden_layer1,w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([64,16]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.random.normal([16]), name = 'bias3')

Hidden_layer3 = tf.nn.selu(tf.matmul(Hidden_layer2,w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([16,13]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.random.normal([13]), name = 'bias4')

Hidden_layer4 = tf.matmul(Hidden_layer3,w4) + b4

w5 = tf.compat.v1.Variable(tf.random.normal([13,3]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([3]), name = 'bias5')

#2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x,w5) + b5) 

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis =1))
# optimizer = tf.train.AdamOptimizer(learning_rate= 0.01).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.000001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)
    
    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)
    
    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    acc = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", acc)


# acc :  0.4864834
# accuracy_score :  0.4864833853497338