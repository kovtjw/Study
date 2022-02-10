import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3],[6,2]]  # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]]       # (6,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

#2. 모델 구성
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
# model = tf.matmul(x,w) + b

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse

#binary_crossentropy 수식
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.04)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 10 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)
        



#4. 평가, 예측
y_pred = tf.cast(hypothesis > 0.5, dtype = tf.float32)  # 실수형으로 변환해 주겠다.
# tf.cast = 텐서를 새로운 형태로 캐스팅하는데 사용한다. 
# 부동 소수점형에서 정수형으로 바군 경우 소수점 버림을 한다.
# boolean형태인 경우 True이면 1, False이면 0
print(sess.run(hypothesis > 0.5, feed_dict = {x:x_data, y:y_data}))

'''
[[False]
 [False]
 [False]
 [ True]
 [ True]
 [ True]]

'''

# accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), dtype = tf.float32))
# pred, acc = sess.run([y_pred, accuracy], feed_dict = {x:x_data, y : y_data})
# print('=====================================================================')
# print('예측값 :', '\n',hy_val)
# print('예측 결과 :', '\n', pred)
# print('accuracy :', acc)

# sess.close()