import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()
a = tf.compat.v1.placeholder(tf.float32)  # 공간을 만든다.
b = tf.compat.v1.placeholder(tf.float32)

adder_node = a + b
print(sess.run(adder_node))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))
add_and_triple = adder_node *3
print(sess.run(add_and_triple, feed_dict = {a:4, b:2}))
# a와 b라는 공간(노드)을 만들어 놓고, 
'''
7.5
[4. 7.]
18.0
'''