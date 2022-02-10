import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[1,2], [2,3],[3,1],[4,3],[5,3][6,2]] #(6,2)
y_data = [[0],[0],[0],[1],[1],[1]] #(6,1)

#2. 모델구성
#실습 시작!!!

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# hypothesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
#model.add(Dense(1, activation='sigmoid'))
# hypothesis = tf.sigmoid(hypothesis)
# model = tf.matmul(x,w)+b

#loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    loss_val, hy_val, _= sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 200 == 0:
        print(epochs, 'loss : ', loss_val, '\n', hy_val)
    
    
    # if epochs % 1 == 0 :
    #     print(epochs, loss_val, w_val, b_val)
'''
#4. 예측
#predict =  x_data*w + b   # predict = model.predict
predict = tf.matmul(x,w) + b

y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})
print("예측 : " , y_predict)

sess.close()

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

mae = mean_absolute_error(y_data, y_predict)
print('mae : ', mae)

# 예측 :  [[ 0.09499058]
#  [-0.14033395]
#  [-0.30428106]
#  [-0.56339806]
#  [-0.7749302 ]
#  [-0.9626697 ]]
# r2스코어 :  -5.378644446485858
# mae :  0.9734339167674383

'''