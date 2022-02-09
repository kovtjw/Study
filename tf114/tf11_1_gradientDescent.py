from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(66)

x_data = [2]
y_data = [1,2,3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# w = tf.compat.v1.Variable(tf.random_normal([1]), name = 'weight')
w = tf.compat.v1.Variable(2, dtype=tf.float32)
hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))
lr = 0.21 
gradient = tf.reduce_mean((x * w - y)*x)
descent = w - lr * gradient    
# w = w-lr*gradient 
update = w.assign(descent)   # tf형에서는 재귀가 안되기 때문에 assign을 사용 함

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(21):
    _, loss_v, w_v = sess.run([update,loss, w], feed_dict={x:x_data, y:y_data})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()

print('===================w history======================')
print(w_history)
print('===================loss history======================')
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()