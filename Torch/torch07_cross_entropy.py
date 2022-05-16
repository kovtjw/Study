from sklearn.datasets import load_breast_cancer, load_iris
import torch
import torch.nn as nn
import torch.optim as optim


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, ' 사용DEVICE :', DEVICE)
# torch :  1.11.0  사용DEVICE : cuda

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

# x = torch.FloatTensor(x)
# # y = torch.FloatTensor(y)
# y = torch.LongTensor(y)

print(x.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)

print(type(x_train),type(y_train))

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

#2. 모델

model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
    # nn.Sigmoid()
  
).to(DEVICE)

#3. 컴파일, 훈련
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()   # CrossEntropyLoss사용할 때는 activation 명시할 필요 없음

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    #model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)

    loss.backward()  #역전파
    optimizer.step()
    return loss.item()

# model.fit
EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch: {}, loss: {:.6f}'.format(epoch, loss))


print('=========================== 평가, 예측 ===============================')
def evaluate(model, criterion, x_test, y_test): # validation
    model.eval()  # 가중치 갱신하지 않겠다.
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()
    
loss = evaluate(model, criterion, x_test, y_test)
print('loss : ', loss)

# y_predict = (model(x_test) >= 0.5).float()  # x_test를 넣었을 때 0.5이상이면, True를 반환 / float()로 하면 0 혹은 1로 반환된다. 
# argmax : 최대값의 위치
y_predict = torch.argmax(model(x_test),1)
print(y_predict[:10])


score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu().detach().numpy(), y_predict.cpu().detach().numpy())

print('accuracy : {:.4f}'.format(score))
