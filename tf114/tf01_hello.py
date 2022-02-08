import tensorflow as tf
print(tf.__version__)


hello = tf.constant('Hello World')  # 문자가 들어가도 상관 없다. 
print(hello)

sess = tf.compat.v1.Session()  # 무조건 만들어야 한다 >> 텐서 머신을 만든다.
print(sess.run(hello))
# Tensor("Const:0", shape=(), dtype=string)
# b'Hello World'

###################################################################################
# tf.constant : 상수 / 고정값 / 한 번 정의해 놓으면 바뀌지 않는다. / 대문자로 표시#
# tf.variable : 변수 / 바뀌는 값 / 소문자로 표시                                  #
# tf.placeholder                                                                  #
###################################################################################
