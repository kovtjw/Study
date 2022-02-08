# 실습
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]

# 위 값들을 이용해서 predict 하기
# x_test라는 placeholder를 생성

# y = wx + b
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
# x_train = [1,2,3]
# y_train = [1,2,3]
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

for step in range(21):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                feed_dict={x_train : [1,2,3], y_train : [1,2,3]})
    if step % 1 == 0:
        # print(step, sess.run(loss), sess.run(w), sess.run(b))
        print(step, loss_val, w_val, b_val)



#4. 예측
test = x_test * w + b
# sess = tf.Session()
print(sess.run(test ,feed_dict = {x_test:[6,7,8]}))
## 입력 값은 placeholder로 
sess.close()

# [4.4951653 5.134803  5.7744403]
# [4.4951653 5.134803  5.7744403]