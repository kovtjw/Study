from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding
from tensorflow.keras.preprocessing import sequence

num_words = 10000
max_len = 500
batch_size = 32


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
print(len(x_train))  
print(len(x_test))   

pad_x_train = sequence.pad_sequences(x_train,maxlen=max_len)
pad_x_test = sequence.pad_sequences(x_test,maxlen=max_len)
print(pad_x_train.shape) 
print(pad_x_test.shape)  

model = Sequential()

model.add(Embedding(num_words,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])
model.summary()

history = model.fit(pad_x_train, y_train,
                    epochs=2,
                    batch_size=128,
                    validation_split=0.2)


import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)
plt. plot(epochs, loss, 'b--', label = 'training loss')
plt. plot(epochs, val_loss, 'r:', label = 'validation loss')
plt.grid()
plt.legend()

plt.figure()
plt. plot(epochs, acc, 'b--', label = 'training accuracy')
plt. plot(epochs, val_acc, 'r:', label = 'validation accuracy')
plt.grid()
plt.legend()

plt.show()

model.evaluate(pad_x_test,y_test)