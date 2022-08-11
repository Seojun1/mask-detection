from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# CNN 모델과 OpenCV의 DNN 모듈 가져오기
facenet = cv2.dnn.readNet('../models/deploy.prototxt', '../models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('../models/mymodel.h5')

img = cv2.imread('../data/without_mask/0.jpg')
h, w = img.shape[:2]

# face-detection을 위한 image preprocessing
blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
facenet.setInput(blob)
dets = facenet.forward()

faces = []

# 이미지 안에 얼굴이 들어있다면 실행, 없다면 Pass
for i in range(dets.shape[2]):
    confidence = dets[0, 0, i, 2]
    if confidence < 0.5:
        continue

    # 얼굴영역의 각 좌표를 구해준다.
    x1 = int(dets[0, 0, i, 3] * w)
    y1 = int(dets[0, 0, i, 4] * h)
    x2 = int(dets[0, 0, i, 5] * w)
    y2 = int(dets[0, 0, i, 6] * h)

    # 얼굴영역만 잘라낸다

    face = img[y1:y2, x1:x2]
    faces.append(face)

    # 잘라낸 얼굴영역 마스크 예측 (이미지 전처리 과정에서 mobileNet 사용)
    for i, face in enumerate(faces):
        face_input = cv2.resize(face, dsize=(64, 64))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        mask, nomask = model.predict(face_input).squeeze()
        cv2.imshow('img', face[:, :, ::-1])
        cv2.waitKey(0)
        if (mask * 100) > 0.5:
            print('마스크 착용중입니다. 마스크 착용 수치:','%.2f%%' % (mask * 100))
        else:
            print('마스크 미착용중인 것 같은데요? 마스크 미착용 수치:','%.2f%%' % (mask * 100))
