import autokeras as ak
import tensorflow as tf

mnist = tf.keras.datasets.mnist
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#2. 모델
model = ak.ImageClassifier(
    overwrite = True,
    max_trials = 2,     # model을 두 번 돌리겠다. 
)

#3. 컴파일, 훈련
model.fit(x_train, y_train, epochs=5)

#4. 평가, 예측
y_pred = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)
model.summary()