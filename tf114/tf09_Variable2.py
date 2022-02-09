import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x = [1,2,3]
y = [3,5,7]

x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])
x_test = tf.placeholder(tf.float32, shape = [None])

w = tf.Variable([0.3])
b = tf.Variable([1.0])


hypothesis = x * w + b

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.17)
train = optimizer.minimize(loss)

# #3-2. 훈련
# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())

# #2. 
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# # bbb = 변수.eval(session = sess)

#3.
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                feed_dict={x_train : [1,2,3], y_train : [3,5,7]})
    if step % 1 == 0:
        # print(step, sess.run(loss), sess.run(w), sess.run(b))
        print(step, loss_val, w_val, b_val)

#4. 평가, 예측
test = x_test * w + b

#1.
# print(sess.run(test ,feed_dict = {x_test:[6,7,8]}))

#2.
# ccc = test.eval(session = sess, feed_dict = {x_test:[6,7,8]})
# print('ccc :', ccc)

#3.
print(test.eval(feed_dict = {x_test:[6,7,8]}))
# print(ddd)?
sess.close()