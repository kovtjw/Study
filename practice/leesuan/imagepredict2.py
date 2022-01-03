# 세이브 한뒤에
# 세이브 한 소스는 주석처리
# 로드해서 처리

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from tensorflow.python.keras.layers.core import Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.3
)

# D:\_data\image\horse-or-human\

train_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/',
    target_size=(50, 50),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=10,
    class_mode='binary',
    subset='training',
    shuffle=True,
)       # Found 719 images belonging to 2 classes.

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/',
    target_size=(50,50),
    batch_size=10,
    class_mode='binary',
    subset='validation'    
)       # Found 308 images belonging to 2 classes.

print(train_generator[0][0].shape, train_generator[0][1].shape)        # (10, 50, 50, 3) (10,)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(50,50,3)))
model.add(MaxPool2D(2))
model.add(Conv2D(16, (2,2)))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M")   # 월일_시분

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
model_path = "".join([filepath, 'k48_2_horse_or_human_IDG_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1, save_best_only= True, filepath = model_path)

start = time.time()
hist = model.fit_generator(train_generator, epochs=100, steps_per_epoch=72,    # steps_per_epoch = 전체 데이터 수 / batch = 160 / 5 = 32
                    validation_data=validation_generator,
                    validation_steps=4, callbacks=[es, mcp]
                    )
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

# plt.plot(epochs, loss, 'r--', label="loss")
# plt.plot(epochs, val_loss, 'r:', label="loss")
# plt.plot(epochs, acc, 'b--', label="acc")
# plt.plot(epochs, val_acc, 'b:', label="val_acc")

#4. 평가, 예측

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/predict/'
sample_image = sample_directory + "123.jpg"

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

print("-- Evaluate --")
scores = model.evaluate_generator(validation_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
validation_generator.reset()
print(validation_generator.class_indices)
# {'hores': 0, 'human': 1}
if(classes[0][0]<=0.5):
    hores = 100 - classes[0][0]*100
    print(f"당신은 {round(hores,2)} % 확률로 horse 입니다")
elif(classes[0][0]>0.5):
    human = classes[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 human 입니다")
else:
    print("ERROR")


# 걸린시간 :  314.649 초
# loss :  0.3621101975440979
# val_loss :  0.5035870671272278
# acc :  0.8331015110015869
# val_acc :  0.824999988079071