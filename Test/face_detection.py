from tensorflow.keras.models import load_model
import os
import cv2
import matplotlib.pyplot as plt

# CNN 모델과 OpenCV의 DNN 모듈을 가져오기
facenet = cv2.dnn.readNet('../models/deploy.prototxt', '../models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('../models/mymodel.h5')

img = cv2.imread('../people.jpg')
h, w = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
facenet.setInput(blob)
dets = facenet.forward()

faces = []

for i in range(dets.shape[2]):
    confidence = dets[0, 0, i, 2]
    if confidence < 0.5:
        continue

    # 얼굴영역의 좌표 저장
    x1 = int(dets[0, 0, i, 3] * w)
    y1 = int(dets[0, 0, i, 4] * h)
    x2 = int(dets[0, 0, i, 5] * w)
    y2 = int(dets[0, 0, i, 6] * h)

    # 얼굴영역만 잘라낸다
    face = img[y1:y2, x1:x2]
    faces.append(face)

    for i, face in enumerate(faces):
        cv2.imshow('img', face[:, :, ::-1])
        cv2.waitKey(0)
