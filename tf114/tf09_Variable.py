import tensorflow as tf
tf.compat.v1.set_random_seed(66)

변수 = tf.Variable(tf.random_normal([1]), name = 'weight')
print(변수)  
# <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref> // shape=(1,) 스칼라가 한 개, input_dim = 1

# 변수 사용의 3가지 방법

#1. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)    # tf형을 사람형으로 변환
print('aaa :', aaa)  # aaa : [0.06524777]
sess.close()

#2. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session = sess) # 변수.eval(session = sess)
print('bbb : ', bbb)            # bbb :  [0.06524777]

#3.
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()      # eval내에 session을 정의하지 않아도 된다. 
print('ccc :', ccc)