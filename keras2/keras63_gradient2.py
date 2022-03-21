import numpy as np

'''
y = x**2 - 4x + 6
미분하면, y' = 2x - 4
'''
f = lambda x: x**2 - 6*x + 6
gradient = lambda x: 2*x-6
x = 10.0       # 초기 값
epochs = 100
learning_rate = 0.25
print('step\t x\t f(x)')
print('{:02d}\t {:6.5}\t {:6.5f}\t'.format(0, x, f(x)))
print('================================================')
for i in range(epochs):
    x = x - learning_rate * gradient(x)
    
    print('{:02d}\t {:6.5}\t {:6.5f}\t'.format(i+1, x, f(x)))