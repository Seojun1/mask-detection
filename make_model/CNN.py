from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Activation, Dropout, Flatten, Dense
import numpy as np
import os
import tensorflow as tf

# 카테고리 지정하기
categories = ["with_mask", "without_mask"]
nb_classes = len(categories)

# 이미지 크기 지정하기
image_w = 64
image_h = 64

# 데이터 열기
X_train, X_test, y_train, y_test = np.load("../npy_files/mynpy.npy", allow_pickle=True)
# 데이터 정규화하기 (0~1 사이로)
X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
print('X_train shape:', X_train.shape)

# CNN 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# tensorboard 변수에 기능을 만들어놓은 것
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../models')

# 모델 구축하기
model.compile(loss='categorical_crossentropy',  # 최적화 함수 지정
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()

# 학습 완료된 모델 저장
model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=[tensorboard])

# 모델 평가하기
score = model.evaluate(X_test, y_test)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc

if not os.path.exists('../models'):
    os.mkdir('../models')
model.save('../models/mymodel.h5')

