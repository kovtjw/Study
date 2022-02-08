import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)   # tensor 연산을 하기 위해서 tf.- 를 명시해주어야 한다.
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)

print(node3)

sess = tf.compat.v1.Session()
print('node1, node2:', sess.run([node1,node2])) # 리스트 해줘야 한다.
print('node3:,', sess.run(node3))