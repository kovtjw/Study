import tensorflow as tf
x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([10,64]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.zeros([64]), name = 'bias1')

Hidden_layer1 = tf.nn.selu(tf.matmul(x,w1) + b1)
layers = tf.nn.dropout(Hidden_layer1, keep_porb = 0.7 ) # 남기고 싶은 %를 말한다.