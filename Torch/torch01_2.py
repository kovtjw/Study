# gpu
# criterion 이슈
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

# gpu는 모델에 넣어주고, 모델에 집어넣는 데이터에 넣어주어야 한다.
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :',DEVICE)


#1. 데이터 (정제된 데이터)
x = np.array([1,2,3])
y = np.array([1,2,3])

# 텐서형 데이터로 변환해줘야 함
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)   # (3,) -> (3,1) 
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   # (3,) -> (3,1)

# unsqueeze( ):

print(x,y)
print(x.shape, y.shape)

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1)) # 데이터 세트에 하나를 넣어서 하나를 얻겠다.
# Dense는 밀집된 레이어의 신경망 구조 

model = nn.Linear(1, 1).to(DEVICE)     # 인풋, 아웃풋 / x의 1, y의 1
# (3,) - > (3, 1) y값도 행렬 형태로 사용해야 한다. 

#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam') 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



# model.fit(x, y, epochs=2500, batch_size=2) 
def train(model, criterion, optimizer, x, y):
    # model.train()        # 훈련모드 
    optimizer.zero_grad()  # 기울기 초기화 : 반드시 넣어야 한다. 
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)
    # 여기까지가 순전파
    loss.backward()       # 기울기 값 계산까지다 
    optimizer.step()      # 가중치 수정
    return loss.item()    # 사람이 보기 좋게 만들어주기


epochs = 500
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch : {}, loss: {}'.format(epoch, loss))
    
print('========================================================')

#4. 평가, 예측 
def evaluate(model, criterion,x, y ):
    model.eval()    # 평가모드
    
    with torch.no_grad():
        predict = model(x)
        loss = criterion(predict, y)
    return loss.item()
    
loss2 = evaluate(model, criterion, x,y)
print('최종 loss : ', loss2)

result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값 :', result)