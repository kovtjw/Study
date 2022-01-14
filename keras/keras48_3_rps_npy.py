import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

x_train = np.load('../_data/_save_npy/keras48_3_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_3_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_3_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_3_test_y.npy')

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])
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

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image



# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = 'D:\\_data\\image\\predict\\'
sample_image = sample_directory + "bo.PNG"

# 샘플 케이스 확인
image_ = plt.imread(str(sample_image))
plt.title("Test Case")
plt.imshow(image_)
plt.axis('Off')
plt.show()

# 샘플케이스 평가
loss, acc = model.evaluate(validation_generator)    # steps=5
#TypeError: 'float' object is not subscriptable
print("Between R.S.P Accuracy : ",str(np.round(acc ,2)*100)+ "%")# 여기서 accuracy는 이 밑의 샘플데이터에 대한 관측치가 아니고 모델 내에서 가위,바위,보를 학습하고 평가한 정확도임

image_ = keras_image.load_img(str(sample_image), target_size=(100, 100))
x = keras_image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /= 255.
# print(x)
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
y_predict = np.argmax(classes)#NDIMS
# print(classes)          # [[0.33294314 0.32337686 0.34368002]]

validation_generator.reset()
print(validation_generator.class_indices)
# class_indices
# {'paper': 0, 'rock': 1, 'scissors': 2}

if(y_predict==0):
    print(classes[0][0]*100, "의 확률로")
    print(" → '보'입니다. " )
elif(y_predict==1):
    print(classes[0][1]*100, "의 확률로")
    print(" → '바위'입니다. ")
elif(y_predict==2):
    print(classes[0][2]*100, "의 확률로")
    print(" → '가위'입니다. ")
else:
    print("ERROR")