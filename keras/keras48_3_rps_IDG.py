import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy

# 1,027 / 10
# 1. 데이터

train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3          
    )                   # set validation split

batch_num = 5

train_generator = train_datagen.flow_from_directory(
    '../_data/image/rps/',
    target_size=(100,100),
    batch_size=batch_num,
    class_mode='categorical',
    subset='training') # 1764

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/rps/', # 756
    target_size=(100,100),
    batch_size=batch_num,
    class_mode='categorical',
    subset='validation')

print(train_generator[0][1].shape)  # (10, 50, 50, 3) (10,)
print(validation_generator[0][1].shape) # (10, 50, 50, 3) (10,)

# np.save('../_data/_save_npy/keras48_3_train_x.npy', arr = train_generator[0][0])
# np.save('../_data/_save_npy/keras48_3_train_y.npy', arr = train_generator[0][1])
# np.save('../_data/_save_npy/keras48_3_test_x.npy', arr = validation_generator[0][0])
# np.save('../_data/_save_npy/keras48_3_test_y.npy', arr = validation_generator[0][1])

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (100,100,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(3,activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])

hist = model.fit_generator(train_generator, epochs = 20, steps_per_epoch = 333, 
                    validation_data = validation_generator,
                    validation_steps = 4,)

model.save('./_save/keras48_3_save.h5')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
import pandas as pd

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
# pic_path = '../_data/image/predict/bo.PNG'
# model_path = './_save/keras48_1_save_weights1111.h5'

# def load_my_image(img_path,show=False):
#     img = image.load_img(img_path, target_size=(50,50))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis = 0)
#     img_tensor /=255.
    
#     if show:
#         plt.imshow(img_tensor[0])    
#         plt.append('off')
#         plt.show()
    
#     return img_tensor

# if __name__ == '__main__':
#     model = load_model(model_path)
#     new_img = load_my_image(pic_path)
#     pred = model.predict(new_img)
#     cat = pred[0][0]*100
#     dog = pred[0][1]*100
    
#     if cat > dog:
#         print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
#     else:
#         print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
