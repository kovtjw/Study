from base64 import b16decode
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

# bias = 행렬의 덧셈
# 입력 값은 placeholder
# zeros는 통상적으로 bias에 넣는다.

# Input Layer
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.random.normal([2,1]), name = 'weight1') # 두 번째 레이어의 노드 수에 맞게 설정
b = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias1')  # 

#2. 모델 구성

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)   # x는 상위 레이어의 아웃풋이다.

#3-1. 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(101):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 10 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)
        
#4. 평가, 예측
y_pred = tf.cast(hypothesis > 0.5, dtype = tf.float32) 
print(sess.run(hypothesis > 0.5, feed_dict = {x:x_data, y:y_data}))

accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), dtype = tf.float32))
pred, acc = sess.run([y_pred, accuracy], feed_dict = {x:x_data, y : y_data})

print('예측 결과 :', '\n', pred)
print('accuracy :', acc)


'''
레이어를 추가하지 않고, 노드 수를 늘리면서 정확도를 높일 수 있었다. >> 겨울 해결!

'''