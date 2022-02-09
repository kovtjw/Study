import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x1_data = [73., 93., 89., 96., 73.]     # 국어
x2_data = [80., 88., 91., 98., 66.]     # 영어
x3_data = [75., 93., 90., 100., 70.]    # 수학
y_data = [152., 185., 180., 196., 142.] # 환산 점수

# x는 (5,3), y는 (5,1) 또는 (5,)
# y = x1 * w1 + x2 * w2 + x3 * w3

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight1')   #  tf.random_normal([1]) shape를 말한다.
w2 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.compat.v1.Variable(tf.random_normal([1]), name = 'bias')

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run([w1,w2,w3]))

#2. 모델
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    _, loss_val, w_val1, w_val2, w_val3, b_val = sess.run([train, loss, w1,w2,w3, b],
                                         feed_dict = {x1 : x1_data, x2 : x2_data, x3 : x3_data,
                                                      y:y_data})
    if epochs % 1 == 0:
        print(epochs, loss_val, w_val1, w_val2, w_val3, b_val)
        
#4. 평가, 예측
y_pred = x1 * w1 + x2 * w2 + x3 * w3 + b
y_pred_data = sess.run(y_pred, feed_dict={x1 : x1_data, x2 : x2_data, x3 : x3_data})
print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data , y_pred_data)
print('r2 :', r2)


