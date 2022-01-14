import numpy as np
import pandas as pd

x_train = np.load('../_data/_save_npy/keras48_1_train_x.npy')
y_train = np.load('../_data/_save_npy/keras48_1_train_y.npy')
x_test = np.load('../_data/_save_npy/keras48_1_test_x.npy')
y_test = np.load('../_data/_save_npy/keras48_1_test_y.npy')


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (50,50,3)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu')) 
model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 

# mcp = ModelCheckpoint(monitor= 'val_loss', mode = 'min', verbose =1, save_best_only=True,
#                       filepath = './_save/keras48_1_save_weights21.h5')
hist = model.fit(x_train, y_train, epochs = 20, batch_size = 32,
                 validation_split= 0.2)  
model.save('./_save/keras48_1_save_11.h5')


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

pic_path = '../_data/image/cat_dog/predict/123.jpg'
model_path = './_save/keras48_1_save_weights1111.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(50,50))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        plt.append('off')
        plt.show()
    
    return img_tensor

if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    cat = pred[0][0]*100
    dog = pred[0][1]*100
    if cat > dog:
        print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
    else:
        print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
        
        