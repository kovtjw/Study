from sklearn.datasets import load_boston
import tensorflow as tf
tf.set_random_seed(42)

#1. 데이터
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(506,1)

# print(x.shape, y.shape)  # (506, 13) (506,)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random.normal([13,1]), name = 'weight')
b = tf.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델 구성
hypothesis = tf.matmul(x, w) + b 

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(200001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                         feed_dict = {x : x_data,
                                                      y:y_data})
    if epochs % 2000== 0:
        print(epochs, loss_val,)
        

#4. 평가, 예측
y_pred = tf.matmul(x, w) + b
y_pred_data = sess.run(y_pred, feed_dict={x : x_data})
# print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data , y_pred_data)
print('r2 :', r2)