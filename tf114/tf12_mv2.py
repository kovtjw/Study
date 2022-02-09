import tensorflow as tf
tf.set_random_seed(66)

#1. 데이터
x_data = [[73, 51,65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [180], [205], [142]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random.normal([3,1]), name = 'weight')  
# (x의 열의 개수와 동일한 행의 갯수가 필요함), y의 열의 개수를 열로
b = tf.Variable(tf.random.normal([1]), name = 'bias')
# bias는 덧셈이기 때문에 shape가 상관이 없다.

# hypothesis = x * w + b 
hypothesis = tf.matmul(x, w) + b  # 행렬 곱으로 변경해 주어야 한다.
#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                         feed_dict = {x : x_data,
                                                      y:y_data})
    if epochs % 20 == 0:
        print(epochs, loss_val,w_val, b_val)
        
#4. 평가, 예측
y_pred = tf.matmul(x, w) + b
y_pred_data = sess.run(y_pred, feed_dict={x : x_data})
print('y_pred_data :',y_pred_data)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data , y_pred_data)
print('r2 :', r2)