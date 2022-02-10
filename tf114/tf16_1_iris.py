import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
tf.set_random_seed(66)

#1. 데이터
datasets = load_iris()
x_data = datasets.data
y_data = datasets.target.reshape(-1,1)
ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()
print(x_data.shape,y_data.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7,random_state=104)

#2. 모델 구성
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.zeros([4, 3]), name = 'weight')
b = tf.Variable(tf.zeros([1,3]), name = 'bias')  # 컬럼에 맞춰서 늘려준다.

#2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) 

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis =1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.06).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)


    # y_acc_test = sess.run(tf.argmax(y_test, 1)) # predict와 맞춰줘야지 accuracy_score이 작동한다.
    # predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    # acc = accuracy_score(y_acc_test, predict)
    # print("accuracy_score : ", acc)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)






# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
    
#     for step in range(2001):
#         _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_data, y:y_data})
#         if step % 200 ==0:
#             print(step, loss_val)
#     results = sess.run(hypothesis, feed_dict = {x : x_data})
#     print(results, sess.run(tf.math.argmax(results, 1)))
#     accuracy =tf.reduce_mean(tf.cast(tf.equal(y_data, results), dtype = tf.float32))
#     print( 'accuracy :', sess.run(accuracy))