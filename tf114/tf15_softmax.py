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
x_pred = [[1,11,7,9]]  # 1,4 > N,4

#2. 모델 구성
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random.normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random.normal([1, 3]), name = 'bias')  # 컬럼에 맞춰서 늘려준다.

#2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)  # tf.nn.softmax
# model.add(Dense(3, activation = 'sofrmax'))

#3-1. 컴파일
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))


loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis =1))
# categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001).minimize(loss)
# train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x:x_data, y:y_data})
        if step % 200 ==0:
            print(step, loss_val)
    # predict 
    results = sess.run(hypothesis, feed_dict = {x : x_pred})
    print(results, sess.run(tf.math.argmax(results, 1)))

# 추가로 할 수 있는 것
# accuracy_score


# # #4. 평가, 예측
# # y_pred = tf.cast(hypothesis > 0.5, dtype = tf.float32)  # 실수형으로 변환해 주겠다.

# # accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), dtype = tf.float32))
# # pred, acc = sess.run([y_pred, accuracy], feed_dict = {x:x_data, y : y_data})
# # print('=====================================================================')
# # print('예측 결과 :', '\n', pred)
# # print('accuracy :', acc)