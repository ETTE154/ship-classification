# 어선 분류 CNN
# 화물선, 어선, 군함 분류

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

#%%

# fish폴더에 있는 이미지 파일을 읽어서 데이터 프레임으로 만들기
# 이미지 파일을 읽어서 numpy 배열로 만들기
# 이미지 파일의 크기를 224x224로 변경

fish_dir = 'ship/fish'
fish_files = os.listdir(fish_dir)

war_dir = 'ship/war'
war_files = os.listdir(war_dir)

cargo_dir = 'ship/cargo'
cargo_files = os.listdir(cargo_dir)

#%%
# 이미지 파일을 읽어서 numpy 배열로 만들기
# 이미지 파일의 크기를 224x224로 변경

def read_img(img_dir, img_files):
    imgs = []
    for file in img_files:
        img = cv2.imread(img_dir + '/' + file)
        img = cv2.resize(img, dsize=(224, 224))
        imgs.append(img)
    return np.array(imgs)

fish_imgs = read_img(fish_dir, fish_files)
war_imgs = read_img(war_dir, war_files)
cargo_imgs = read_img(cargo_dir, cargo_files)

# 파일 출력
print(fish_imgs.shape)
print(war_imgs.shape)
print(cargo_imgs.shape)

#%%

# 배열의 첫번째 이미지 출력
plt.imshow(fish_imgs[0])
plt.show()

plt.imshow(war_imgs[0])
plt.show()

plt.imshow(cargo_imgs[0])
plt.show()

#%%
# 군함, 화물선, 어선 별로 레이블을 만들기
# 0: 군함, 1: 화물선, 2: 어선

fish_labels = np.zeros(len(fish_files))
war_labels = np.ones(len(war_files))
cargo_labels = np.full(len(cargo_files), 2)

fish_labels.shape, war_labels.shape, cargo_labels.shape
#%%
# 데이터 증식

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
# %%
# 이미지 데이터와 레이블을 합치기
imgs = np.concatenate([fish_imgs, war_imgs, cargo_imgs])
labels = np.concatenate([fish_labels, war_labels, cargo_labels])

imgs.shape, labels.shape
# %%

# 이미지 데이터와 레이블을 합치기
imgs = np.concatenate([fish_imgs, war_imgs, cargo_imgs])
labels = np.concatenate([fish_labels, war_labels, cargo_labels])

#%%

# 학습데이타와 테스트 데이터로 나누기
from sklearn.model_selection import train_test_split

train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.2, random_state=42)

train_imgs.shape, test_imgs.shape, train_labels.shape, test_labels.shape

# %%

# 데이터 증식
datagen.fit(train_imgs)
# 증식된 데이터 확인
for x, y in datagen.flow(train_imgs, train_labels, batch_size=9):
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x[i])
    plt.show()
    break

# %%

# 나눠진 데이터 확인
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_imgs[i])
plt.show()

#%%
# 이미지 데이터를 0~1 사이의 값으로 변경
train_imgs_1 = train_imgs / 255.0
test_imgs_1 = test_imgs / 255.0

#%%
# 변경된 데이터 확인
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_imgs_1[i])
plt.show()
# %%
# 모델 생성
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# dropout 비율 0.2
dropout_rate = 0.2

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(dropout_rate))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

# %%
# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
# 모델 학습
# 100번 반복
# 20%는 검증데이터로 사용
# 10번마다 학습 결과 출력
# 성능 향상이 없으면 학습 중단
# 최적의 데이터 저장
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint('ship_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

hist = model.fit_generator(datagen.flow(train_imgs_1, train_labels, batch_size=32), steps_per_epoch=len(train_imgs_1)/32, epochs=100, validation_data=(test_imgs_1, test_labels), callbacks=[checkpoint, early])

# %%

#   학습 결과 확인
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], 'b-', label='loss')
plt.plot(hist.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()    

plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], 'g-', label='accuracy')
plt.plot(hist.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()
# %%
