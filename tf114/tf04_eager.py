import tensorflow as tf
import os
print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.executing_eagerly())

tf.compat.v1.disable_eager_execution()   

print(tf.executing_eagerly())

hello = tf.constant('Hello World')
sess = tf.compat.v1.Session()
print(sess.run(hello))
'''
- tensor2 환경에서 tensor1을 쓰는 법
print(tf.executing_eagerly())  > True >> 즉시 실행 모드 확인 
tf.compat.v1.disable_eager_execution()  > 즉시 실행 모드 off
print(tf.executing_eagerly()) > 즉시 실행 모드 확인
tensor2에서 sess 사용 가능 
'''