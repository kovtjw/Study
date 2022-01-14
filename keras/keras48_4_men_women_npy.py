import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

x_train = np.load('../_data/_save_npy/keras48_4_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_4_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_4_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_4_test_y.npy')

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (100,100,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32,
                 validation_split= 0.2,callbacks=[es])  

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

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
    print(f"당신은 {round(hores,2)} % 확률로 남자 입니다")
elif(classes[0][0]>0.5):
    human = classes[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 여자 입니다")
else:
    print("ERROR")