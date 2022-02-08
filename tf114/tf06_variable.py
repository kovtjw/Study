import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype = tf.float32)

init = tf.global_variables_initializer()  
# 전체 변수를 사용할 수 있게 초기화 함 / 허가해준다는 느낌
sess.run(init)

print(sess.run(x))
