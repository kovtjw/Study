from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(569,1)

# print(x_data.shape, y_data.shape)  # (569, 30) (569,)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([30,1]), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')

#2. 모델 구성
hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate= 1e-9)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(5001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 100 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)
        

#4. 평가, 예측
y_pred = tf.cast(hypothesis > 0.5, dtype = tf.float32)  # 실수형으로 변환해 주겠다.

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), dtype = tf.float32))
pred, acc = sess.run([y_pred, accuracy], feed_dict = {x:x_data, y : y_data})
print('=====================================================================')
print('예측값 :', '\n',hy_val)
print('예측 결과 :', '\n', pred)
print('accuracy :', acc)