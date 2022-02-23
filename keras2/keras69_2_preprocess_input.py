from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
img_path = '../_data/boy.jpg'    
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print('=====================image.img_to_array(img)============================')
# print(x, '\n', x.shape)  # (224, 224, 3)   

x = np.expand_dims(x, axis=0)
print('=====================np.expand_dims(x, axis=0)============================')
# print(x, '\n', x.shape)  #  (1, 224, 224, 3)

x = preprocess_input(x)    # 전이학습에 사용 할 모델에 가장 최적의 스케일링이 적용되어 있다. 
print('=====================preprocess_input(x)============================')
# print(x, '\n', x.shape)

preds = model.predict(x)
# print(np.argmax(preds)) # 207
# print(preds, '\n', preds.shape)  #  (1, 1000)
print('결과는?? :', decode_predictions(preds, top = 10)[0])  # 위에서부터 5개 