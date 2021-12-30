from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.text import one_hot

text1 = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'   
text2 = '나는 매우 매우 잘생긴 지구용사 태권브이'
token = Tokenizer()
token.fit_on_texts([text1,text2])

print(token.word_index)    
# {'매우': 1, '나는': 2, '진짜': 3, '마구': 4, '맛있는': 5, '밥을': 6, 
#  '먹었다': 7, '잘생긴': 8, '지구용사': 9, '태권브이': 10}

# text3 = text1+text2
# x = token.texts_to_sequences([text3]) # 텍스트를 수치화 시킴 >> 숫자가 큰 것에 더 가중치를 줄 수 있는 문제점 >>> 원-핫-인-코-딩
# print(x)   # [[3, 1, 4, 5, 6, 1, 2, 2, 7]] >> 리스트 형태  

x = token.texts_to_sequences([text1,text2]) # 텍스트를 수치화 시킴 >> 숫자가 큰 것에 더 가중치를 줄 수 있는 문제점 >>> 원-핫-인-코-딩
print(x) 

x = x[0] + x[1]          # (9,) + (6,) = (15,)
print(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)                        # 리스트의 크기는 len으로 봐야한다.
print('word_size:', word_size, '가지의 단어의 종류')     # 10
print(x)

x = to_categorical(x)
print(x)
print(x.shape) # (15, 11)

# 문제점 

