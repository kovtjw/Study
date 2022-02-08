from numpy import gradient
gradient = lambda x: 2*x -4
# lambda 함수 :앞에 input을 넣는다.  //  변수 = lambda 인풋 값 : 아웃풋

def gradient2(x):
    return 2*x-4

x = 3
print(gradient(x))
print(gradient2(x))