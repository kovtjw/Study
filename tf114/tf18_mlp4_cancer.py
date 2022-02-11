from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(104)
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(569,1)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.8,random_state=66)

# print(x_data.shape, y_data.shape)  # (569, 30) (569,)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.zeros([30,64]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([64]), name = 'bias1')

Hidden_layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.uniform([64,32]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.random.uniform([32]), name = 'bias2')

Hidden_layer2 = tf.matmul(Hidden_layer1,w2) + b2

w3 = tf.compat.v1.Variable(tf.zeros([32,16]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.zeros([16]), name = 'bias3')

Hidden_layer3 = tf.matmul(Hidden_layer2,w3) + b3

w4 = tf.compat.v1.Variable(tf.random.uniform([16,30]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.zeros([30]), name = 'bias4')

Hidden_layer4 = tf.sigmoid(matmul(Hidden_layer3,w4) + b4)

w5 = tf.compat.v1.Variable(tf.random.normal([30,2]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.random.normal([2]), name = 'bias5')

#2. 모델 구성
hypothesis = tf.sigmoid(tf.matmul(x,w5) + b5)

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.00000001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(10001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_train, y:y_train})
    
    if epochs % 100 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)
        

#4. 평가, 예측
y_pred = tf.cast(hypothesis > 0.5, dtype = tf.float32)  # 실수형으로 변환해 주겠다.

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), dtype = tf.float64))
pred, acc = sess.run([y_pred, accuracy], feed_dict = {x:x_test, y : y_test})
print('예측 결과 :', '\n', pred)
print('accuracy :', acc)