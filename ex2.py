import torch
import torchvision.models as models

#학습한 매개변수를 state_dict(내부상태사전)에 저장
model = models.vgg16(weights='IMAGENT1K_V1')  #사전학습된 VGG16모델 불러오기
torch.save(model.state_dict(), 'model_weights.pth') #모델의 학습된 가중치만 저장


model = models.vgg16()   #weights지정하지 않으므로 학습되지 않은 모델 생성
model.load_state_dict(torch.load('model_weights.pth'))   #불러온 가중치 적용
model.eval()  #모델을 평가모드(test)

torch.save(model, 'model.pth')   #모델 전체 저장(가중치+구조)
model = torch.load('model.pth')    #모델 다시 가져와 사용 가능하도록