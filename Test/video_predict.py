from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

facenet = cv2.dnn.readNet('../models/deploy.prototxt', '../models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('../models/mymodel.h5')

# 웹캠으로 설정
cap = cv2.VideoCapture(0)
ret, img = cap.read()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('../result/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]


    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    facenet.setInput(blob)
    dets = facenet.forward()

    # 테스트할 웹캠
    result_img = img.copy()

    # 이미지 안에 얼굴이 들어있다면 실행, 없다면 Pass
    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        # 얼굴영역만 잘라낸다
        face = img[y1:y2, x1:x2]

        # 잘라낸 얼굴을 predict 할 수 있도록 변형
        face_input = cv2.resize(face, dsize=(64, 64))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        # 예측
        mask, nomask = model.predict(face_input).squeeze()

        # 마스크 쓴 상태와 안쓴 상태에 따라 화면에 띄어질 내용 디자인
        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)

        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)


        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)


    # 웹캠 띄우기
    cv2.imshow('result', result_img)
    if cv2.waitKey(1) == ord('q'):
        break
out.release()
cap.release()
