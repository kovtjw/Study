from tensorflow.keras.layers import SimpleRNN,Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# model = Sequential()
# model.add(Embedding(10000,32))
# model.add(SimpleRNN(32))
# model.summary()

# model = Sequential()
# model.add(Embedding(10000,32))
# model.add(SimpleRNN(32, return_sequences=True))
# model.add(SimpleRNN(32, return_sequences=True))
# model.summary()

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

num_words = 10000
max_len = 500
batch_size = 32

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=num_words)
print(len(input_train))  # 25000
print(len(input_test))   # 25000

input_train = sequence.pad_sequences(input_train,maxlen=max_len)
input_test = sequence.pad_sequences(input_test,maxlen=max_len)
print(input_train.shape) #(25000, 500)
print(input_test.shape)  #(25000, 500)


model = Sequential()

model.add(Embedding(num_words,32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['acc'])
model.summary()

history = model.fit(input_train,y_train,
                    epochs=3,
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

model.evaluate(input_test,y_test)