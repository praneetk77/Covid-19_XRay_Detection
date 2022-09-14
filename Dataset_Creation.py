import pandas as pd
import os
import shutil

# creating dataset for positive samples
FILE_PATH = "covid-chestxray-dataset/metadata.csv"
IMAGES_PATH = "covid-chestxray-dataset/images"

df = pd.read_csv(FILE_PATH)
print(df.shape)
print(df.head())

# TARGET_DIR = "Dataset/Covid"
#
# if not os.path.exists(TARGET_DIR):
#     os.mkdir(TARGET_DIR)
#     print("Covid folder created")
#
# cnt = 0
# for (i,row) in df.iterrows():
#     if row["finding"]=="Pneumonia/Viral/COVID-19" and row["view"]=="PA":
#         filename = row["filename"]
#         image_path = os.path.join(IMAGES_PATH,filename)
#         image_copy_path = os.path.join(TARGET_DIR, filename)
#         shutil.copy2(image_path, image_copy_path)
#         print("Moving image ",cnt)
#         cnt += 1
# print(cnt)
#
# # sampling of images from kaggle (normal x-rays)
# import random
# KAGGLE_FILE_PATH = "chest_xray/train/NORMAL"
# TARGET_NORMAL_DIR = "Dataset/Normal"
#
# image_names = os.listdir(KAGGLE_FILE_PATH)
# random.shuffle(image_names)
#
# for i in range(196):
#     image_name = image_names[i]
#     image_path = os.path.join(KAGGLE_FILE_PATH, image_name)
#
#     target_path = os.path.join(TARGET_NORMAL_DIR, image_name)
#
#     shutil.copy2(image_path, target_path)
#     print("Copying image ", i)
#
# #separating training and validation datasets
# covid_image_names = os.listdir("Dataset/Covid")
# random.shuffle(covid_image_names)
#
# normal_image_names = os.listdir("Dataset/Normal")
# random.shuffle(normal_image_names)
#
# for i in range(20):
#     covid_image_name = covid_image_names[i]
#     covid_image_path = os.path.join("Dataset/Covid", covid_image_name)
#     normal_image_name = normal_image_names[i]
#     normal_image_path = os.path.join("Dataset/Normal", normal_image_name)
#
#     covid_target_path = os.path.join("Dataset/Val/Covid", covid_image_name)
#     normal_target_path = os.path.join("Dataset/Val/Normal", normal_image_name)
#
#     shutil.move(covid_image_path, covid_target_path)
#     shutil.move(normal_image_path, normal_target_path)

# PREPROCESSING COMPLETE

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.preprocessing import *

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

#Training from scratch

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_dataset = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('Dataset/Train', target_size=(224,224), batch_size=32, class_mode='binary')

validation_generator = test_dataset.flow_from_directory('Dataset/Val', target_size=(224,224), batch_size=32, class_mode='binary')

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2
)









