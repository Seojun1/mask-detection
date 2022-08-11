import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

categories = ["with_mask", "without_mask"]

model = load_model('../models/mymodel.h5')

# 적용해볼 이미지
test_image = '../data/without_mask/0.jpg'

# 이미지 resize
img = Image.open(test_image)
img = img.convert("RGB")
img = img.resize((64, 64))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 256
X = X.reshape(-1, 64, 64, 3)

# 예측
pred = model.predict(X)
result = [np.argmax(value) for value in pred]
if categories[result[0]]:
    print('마스크 미착용중입니다.')
else:
    print('마스크 착용중입니다.')