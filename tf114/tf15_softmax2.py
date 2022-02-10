import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
# (8,4)
y_data = [[0,0,1],      # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],      # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],      # 0
          [1,0,0]]
# (8,3)

#2. 모델 구성
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random.normal([1, 3]), name = 'bias')  # 컬럼에 맞춰서 늘려준다.

#2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) 

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis =1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.08).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_data, y:y_data})
        if step % 200 ==0:
            print(step, loss_val)
    # results = sess.run(hypothesis, feed_dict = {x : x_data})
    # print(results, sess.run(tf.math.argmax(results, 1)))
    # accuracy =tf.reduce_mean(tf.cast(tf.equal(y_data, results), dtype = tf.float32))
    # print( 'accuracy :', sess.run(accuracy))
    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)
    
# accuracy : 0.6666667