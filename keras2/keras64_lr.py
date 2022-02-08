x = 10         # 임의로 바꾸기
y = 10         # 목표값
w = 0.5         # 가중치 초기값
lr = 0.001
epochs = 1000

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2    # mse
    
    # 가중치와 epoch도 넣어서 아래 print를 수정
    print('loss :', round(loss,4), '\tPredict :', round(predict,4), '\tepoch :', round(epochs,4), '\tw :', round(w,4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr