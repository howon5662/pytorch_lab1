import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#dataset 지정
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#뉴럴네트워크 생성
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

#하이퍼파라미터
learning_rate = 1e-3
batch_size = 64
epochs = 5    #최적화 단계의 각 반복

#손실함수 초기화
loss_fn = nn.CrossEntropyLoss() #nn.NLLLoss(음의로그우도)+nn.LogSoftmax

#optimizer(하이퍼파라미터 이용)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)  #SGD


#모델 학습
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader): #예측값과 손실(loss)계산
        pred = model(X)
        loss = loss_fn(pred, y)

        #역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*batch_size + len(X)
            print(f"lossL {loss:>7f}[{current:>5d}/{size:>5d}")

#모델 테스트
def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)   #테스트 데이터셋의 전체샘플개수 저장
    num_batches = len(dataloader)    #미니배치 개수저장
    test_loss, correct = 0,0         #손실,맞은개수 저장하는 변수

    with torch.no_grad():  #그래디언트 계산하지 않도록/테스트과정에서는 모델업데이트 필요없음
        for X,y in dataloader:
            pred = model(X)   #x값넣고 예측값(pred)계산(logits)
            test_loss += loss_fn(pred, y).item()   #손실계산
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")


#손실함수 초기화
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n----------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")