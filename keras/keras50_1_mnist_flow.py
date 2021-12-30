# 훈련데이터 10만개로 증폭
# 완료 후 기존 모델과 비교
# save_dir도 temp에 넣고
# 증폭 데이터는 temp에 저장 후 훈련 끝난 후 결과 확인 후 삭제


from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    #vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    #shear_range = 0.7,
    fill_mode= 'nearest')

# 증폭 데이터 생성
augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()  
y_augmented = y_train[randidx].copy() 

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

xy_train = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle= False) 

# xy_test = test_datagen.flow(x_test,y_test,batch_size=32)

x = np.concatenate((x_train, xy_train[0][0]))  # (100000, 28, 28, 1)
y = np.concatenate((y_train, xy_train[0][1]))

# print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape = (28,28,1)))
model.add(Conv2D(64,(2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x, y, epochs=10, steps_per_epoch=1000)


#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict_generator(x_test)

# print(y_test)
# print(y_test.shape, y_predict.shape)     # (10000, 10) (10000, 10)

y_predict=np.argmax(y_predict, axis=1)
# y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score   # accuracy_score 분류에서 사용
a = accuracy_score(y_test, y_predict)
print('acc score:', a)
