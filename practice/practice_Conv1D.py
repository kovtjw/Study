from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, Embedding, Conv1D,MaxPooling1D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
num_words = 10000
max_len = 500
batch_size = 32

(input_train,y_train), (input_test,y_test) = imdb.load_data(num_words=num_words)
print(len(input_train)) # 25000
print(len(input_test))  # 25000

pad_x_train = pad_sequences(input_train,maxlen= max_len)
pad_x_test = pad_sequences(input_test,maxlen= max_len)

print(pad_x_train.shape) # (25000, 500)
print(pad_x_test.shape)  # (25000, 500)

def build_model():
    model = Sequential()
    
    model.add(Embedding(input_dim = num_words, output_dim=32,
                        input_length=max_len))
    model.add(Conv1D(32,7,activation='relu'))
    model.add(MaxPooling1D(7))
    model.add(Conv1D(32,5,activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(optimizer=RMSprop(learning_rate=1e-4),
                  loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# 학습
history =  model.fit(pad_x_train,y_train,
                     batch_size = 128, epochs =10,
                     validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc=history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs =range(1,len(loss)+1)

plt.plot(epochs, loss, 'b--', label = 'training loss')
plt.plot(epochs, val_loss, 'r:', label = 'validation loss')
plt.grid()
plt.legend()

plt.figure()
plt.plot(epochs, acc, 'b--', label = 'training accuracy')
plt.plot(epochs, val_acc, 'r:', label = 'validation accuracy')
plt.grid()
plt.legend()