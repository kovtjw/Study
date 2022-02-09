import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]

x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])
x_test = tf.placeholder(tf.float32, shape = [None])

# w = tf.Variable(1, dtype=tf.float32)
# b = tf.Variable(1, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w)) 

#2. 모델 구성
hypothesis = x_train * w + b    # y = wx + b
# hypothesis = y 라고 생각하면 됨

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))
# mse // square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# model.compili(loss = 'mse',optimizer = 'sgd')

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())  # w, b 변수 초기화

loss_val_list = []
w_val_list = []

for step in range(2001):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                feed_dict={x_train : x_train_data, y_train : y_train_data})
    if step % 10 == 0:
        # print(step, sess.run(loss), sess.run(w), sess.run(b))
        print(step, loss_val, w_val, b_val)
    
    loss_val_list.append(loss_val)
    w_val_list.append(w_val)

#4. 평가, 예측
test = x_test * w + b
# sess = tf.Session()
print(sess.run(test ,feed_dict = {x_test:[6,7,8]}))

sess.close()

import matplotlib.pyplot as plt
plt.plot(loss_val_list[100:])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()