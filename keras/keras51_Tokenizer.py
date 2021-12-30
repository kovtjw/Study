from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'   

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)   
# {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7} >> 반복이 많이 된 순서대로 앞쪽으로 온다.
# 어절 = 띄어쓰기 단위 = Token
# 수치화 가능 

x = token.texts_to_sequences([text]) # 텍스트를 수치화 시킴 >> 숫자가 큰 것에 더 가중치를 줄 수 있는 문제점 >>> 원-핫-인-코-딩
print(x)   # [[3, 1, 4, 5, 6, 1, 2, 2, 7]] >> 리스트 형태  

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print('word_size:', word_size, '가지의 단어의 종류')     # 7

x = to_categorical(x)
print(x)
print(x.shape) # (1, 9, 8)      / 9 = 어절 / 


