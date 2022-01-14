import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras.preprocessing import image as keras_image


# 1,027 / 10
# 1. 데이터

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    #vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    #shear_range = 0.7,
    validation_split=0.3,
    fill_mode= 'nearest')                   # set validation split

train_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/horse-or-human/',
    target_size=(50,50),
    batch_size=5,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/horse-or-human/', # same directory as training data
    target_size=(50,50),
    batch_size=5,
    class_mode='binary',
    subset='validation') # set as validation data

print(train_generator[0][0].shape)  # 719
print(validation_generator[0][0].shape) # 308

# # print(xy_train[0][0].shape, xy_train[0][1].shape)  # (10, 50, 50, 3) (10,)

# # np.save('./_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
# # np.save('./_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
# # np.save('./_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
# # np.save('./_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])


# # 2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

# model = Sequential()
# model.add(Conv2D(32, (2,2), input_shape=(150,150,3)))
# model.add(MaxPool2D(2))
# model.add(Conv2D(16, (2,2)))
# model.add(MaxPool2D(2))
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1, activation='sigmoid'))

# # 3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])

# hist = model.fit_generator(train_generator, epochs = 30, steps_per_epoch = 360, 
#                     validation_data = validation_generator,
#                     validation_steps = 4,)

# # model.save('./_save/keras48_2_save.h5')


# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss:', loss[-1])
# print('val_loss:', val_loss[-1])
# print('acc:', acc[-1])
# print('val_acc:',val_acc [-1])


# #4. 평가, 예측

# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image

# # 샘플 케이스 경로지정
# #Found 1 images belonging to 1 classes.
# sample_directory = '../_data/image/predict/'
# sample_image = sample_directory + "123.jpg"

# # 샘플 케이스 확인
# # image_ = plt.imread(str(sample_image))
# # plt.title("Test Case")
# # plt.imshow(image_)
# # plt.axis('Off')
# # plt.show()

# print("-- Evaluate --")
# scores = model.evaluate_generator(validation_generator, steps=10)
# print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# print("-- Predict --")
# image_ = image.load_img(str(sample_image), target_size=(150, 150))
# x = image.img_to_array(image_)
# x = np.expand_dims(x, axis=0)
# x /=255.
# images = np.vstack([x])
# classes = model.predict(images, batch_size=2)
# # y_predict = np.argmax(classes)#NDIMS

# print(classes)
# validation_generator.reset()
# print(validation_generator.class_indices)
# # {'hores': 0, 'human': 1}
# if(classes[0][0]<=0.5):
#     hores = 100 - classes[0][0]*100
#     print(f"당신은 {round(hores,2)} % 확률로 horse 입니다")
# elif(classes[0][0]>0.5):
#     human = classes[0][0]*100
#     print(f"당신은 {round(human,2)} % 확률로 human 입니다")
# else:
#     print("ERROR")










# import numpy as np
# import pandas as pd

# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt

# # pic_path = '../_data/image/cat_dog/predict/123.jpg'
# # model_path = './_save/keras48_2_save.h5'

# # def load_my_image(img_path,show=False):
# #     img = image.load_img(img_path, target_size=(100,100))
# #     img_tensor = image.img_to_array(img)
# #     img_tensor = np.expand_dims(img_tensor, axis = 0)
# #     img_tensor /=255.
    
# #     if show:
# #         plt.imshow(img_tensor[0])    
# #         plt.append('off')
# #         plt.show()
    
# #     return img_tensor

# # if __name__ == '__main__':
# #     model = load_model(model_path)
# #     new_img = load_my_image(pic_path)
# #     pred = model.predict(new_img)
# #     horse = pred[0][0]*100
# #     human = pred[0][1]*100
# #     if horse > human:
# #         print(f"당신은 {round(horse,2)} % 확률로 horse 입니다")
# #     else:
# #         print(f"당신은 {round(human,2)} % 확률로 human 입니다")


# # '''
# # loss: 3.5137249478666004e-20
# # val_loss: 0.0
# # acc: 1.0
# # val_acc: 1.0
# # '''

# # 샘플 케이스 경로지정
# #Found 1 images belonging to 1 classes.

# sample_image = '../_data/image/cat_dog/predict/123.jpg'

# # 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# # 샘플케이스 평가
# loss, acc = model.evaluate(validation_generator)    # steps=5
# #TypeError: 'float' object is not subscriptable

# image_ = keras_image.load_img(str(sample_image), target_size=(100, 100))
# x = keras_image.img_to_array(image_)
# x = np.expand_dims(x, axis=0)
# x /= 255.
# # print(x)
# images = np.vstack([x])
# classes = model.predict(images, batch_size=32)
# y_predict = np.argmax(classes)#NDIMS
# # print(classes)          # [[0.33294314 0.32337686 0.34368002]]

# validation_generator.reset()
# print(validation_generator.class_indices)
# # class_indices
# # {'paper': 0, 'rock': 1, 'scissors': 2}

# if(y_predict==0):
#     print(classes[0][0]*100, "의 확률로")
#     print(" → '말'입니다. " )
# elif(y_predict==1):
#     print(classes[0][1]*100, "의 확률로")
#     print(" → '사람'입니다. ")
# else:
#     print("ERROR")

# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.backend import binary_crossentropy

# # 1,027 / 10
# # 1. 데이터

# train_datagen = ImageDataGenerator(
#     rescale = 1./255,              
#     horizontal_flip = True,        
#     vertical_flip= True,           
#     width_shift_range = 0.1,       
#     height_shift_range= 0.1,       
#     rotation_range= 5,
#     zoom_range = 1.2,              
#     shear_range=0.7,
#     fill_mode = 'nearest',
#     validation_split=0.3          
#     )                   # set validation split

# train_generator = train_datagen.flow_from_directory(
#     '../_data/image/horse-or-human/',
#     target_size=(100,100),
#     batch_size=10,
#     class_mode='binary',
#     subset='training') # set as training data

# validation_generator = train_datagen.flow_from_directory(
#     '../_data/image/horse-or-human/', # same directory as training data
#     target_size=(100,100),
#     batch_size=10,
#     class_mode='binary',
#     subset='validation') # set as validation data

# print(train_generator[0][0].shape)  # 719
# print(validation_generator[0][0].shape) # 308


# # np.save('./_save_npy/keras48_2_train_x.npy', arr = train_generator[0][0])
# # np.save('./_save_npy/keras48_2_train_y.npy', arr = train_generator[0][1])
# # np.save('./_save_npy/keras48_2_test_x.npy', arr = validation_generator[0][0])
# # np.save('./_save_npy/keras48_2_test_y.npy', arr = validation_generator[0][1])

# # print(xy_train[0][0].shape, xy_train[0][1].shape)  # (10, 50, 50, 3) (10,)

# # 2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(32,(2,2), input_shape = (50,50,3)))
# model.add(Flatten())
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu')) 
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# # 3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])

# hist = model.fit_generator(train_generator, epochs = 3, steps_per_epoch = 72, 
#                     validation_data = validation_generator,
#                     validation_steps = 4,)

# model.save('./_save/keras48_2_save.h5')

# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss:', loss[-1])
# print('val_loss:', val_loss[-1])
# print('acc:', acc[-1])
# print('val_acc:',val_acc [-1])


# #############prtdict##############

# import numpy as np
# import pandas as pd

# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt

# pic_path = './_data/image/predict/123.jpg'
# model_path = './_save/keras48_2_save.h5'

# def load_my_image(img_path,show=False):
#     img = image.load_img(img_path, target_size=(100,100))
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
#         print(f"당신은 {round(cat,2)} % 확률로 horse니다")
#     else:
#         print(f"당신은 {round(dog,2)} % 확률로 human입니다")