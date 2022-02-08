# y = wx + b
import tensorflow as tf
tf.set_random_seed(66)
x_train = [1,2,3]
y_train = [1,2,3]
w = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)
#2. 모델 구성
hypothesis = x_train * w + b    # y = wx + b
# hypothesis = y 라고 생각하면 됨
#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))
# mse // square = 제곱, reduce_mean = 평균
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)
# model.compili(loss = 'mse',optimizer = 'sgd')

#3-2. 훈련
# sess = tf.compat.v1.Session()

with tf.compat.v1.Session() as sess:
    
    sess.run(tf.compat.v1.global_variables_initializer())  # w, b 변수 초기화

    for step in range(21):
        sess.run(train)
        if step % 1 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()


