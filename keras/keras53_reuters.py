from tensorflow.keras.datasets import reuters
import numpy as np 
import pandas as pd

(x_train,y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, test_split = 0.2)

print(len(x_train))     # 8982
print(len(x_test))

print(np.unique(y_train))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape)  # (8982,) (8982,)
print(len(x_train[0]), len(x_train[1]))  # 87 56
print(type(x_train[0]), type(x_train[1]))   # <class 'list'> <class 'list'>

# print('뉴스 기사의 최대길이:', max(len(x_train)))    # TypeError: 'int' object is not iterable
print('뉴스 기사의 최대 길이:', max(len(i) for i in x_train))   # 2376
print('뉴스 기사의 평균 길이:', sum(map(len, x_train)) / len(x_train))  # 145.5398574927633

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, padding='pre', maxlen = 100, truncating='pre')    # maxlen = 안써도 됨 ㅋ
print(x_train.shape) # (8982, 2376)      # 최대 값만큼 변한다.
x_test = pad_sequences(x_test, padding='pre', maxlen = 100, truncating='pre')

print(y_train.shape, y_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)  # (8982, 46) (2246, 46)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()
model.add(Embedding(input_dim = 1000, output_dim=10, input_length=100))                  # 단어 사전의 개수
model.add(LSTM(32))
model.add(Dense(64,activation='relu'))
model.add(Dense(32))                                                  
model.add(Dense(16))                                                                                                    
model.add(Dense(46, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=2, batch_size=32)

#4. 평가, 예측
acc = model.evaluate(x_test,y_test)[1]
print('acc:', acc)


# reuters = 뉴스 
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
# >> 뉴스 카테고리


######################################추가######################################
word_to_index = reuters.get_word_index()
# print(word_to_index)
# print(sorted(word_to_index.items()))     # 키 위주로 나온다.
import operator
print(sorted(word_to_index.items(),key = operator.itemgetter(1)))   # key = operator.itemgetter(0)) >> key를 의미




index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key
    
for index, token in enumerate(('<pad>','<sos','<unk>')):
    index_to_word[index] = token
    
print(' '.join([index_to_word[index] for index in x_train[0]]))