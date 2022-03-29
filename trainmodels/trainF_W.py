# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# train_data_dir='threefingerdata/train'
# validation_data_dir='threefingerdata/test'
#
# train = train_datagen =ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# test = ImageDataGenerator(rescale=1 / 255)
#
# train_dataset = train.flow_from_directory(train_data_dir,
#                                           target_size=(350, 350),
#                                           batch_size=10,
#                                           class_mode='binary')
#
# test_dataset = test.flow_from_directory(validation_data_dir,
#                                         target_size=(350, 350),
#                                         batch_size=10,
#                                         class_mode='binary')
# # test_dataset.class_indices
#
# model = keras.Sequential()
#
# # Convolutional layer and maxpool layer 1
# model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(350,350,3)))
# model.add(keras.layers.MaxPool2D(2,2))
#
# # Convolutional layer and maxpool layer 2
# model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
# model.add(keras.layers.MaxPool2D(2,2))
#
# # Convolutional layer and maxpool layer 3
# model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
# model.add(keras.layers.MaxPool2D(2,2))
#
# # Convolutional layer and maxpool layer 4
# model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
# model.add(keras.layers.MaxPool2D(2,2))
#
# # This layer flattens the resulting image array to 1D array
# model.add(keras.layers.Flatten())
#
# # Hidden layer with 512 neurons and Rectified Linear Unit activation function
# # model.add(keras.layers.Dense(512,activation='relu'))
#
# # Output layer with single neuron which gives 0 for Cat or 1 for Dog
# #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
# model.add(keras.layers.Dense(1,activation='sigmoid'))
#
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#
# model.fit_generator(train_dataset,
#                     steps_per_epoch=2000//5,
#                     epochs=5,
#                     validation_data=1002//5
#                     )
#
# model_json = model.to_json()
# with open("F_W.json", "w") as json_file:
#     json_file.write(model_json)
# print('Model Saved')
# model.save_weights('F_W.h5')
# print('Weights saved')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image

# dimension of images
img_width, img_height = 350, 350

train_data_dir='threefingerdata/train'
validation_data_dir='threefingerdata/test'
nb_train_samples= 1500
nb_validation_samples= 700
epochs = 5
batch_size = 5

if K.image_data_format() == 'channels_first':
    input_shape =(3, img_width, img_height)
else:
    input_shape =(img_width,img_height, 3)

train_datagen =ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    color_mode='rgb'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    color_mode='rgb'
)

# Building cnn model
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size
)

model_json = model.to_json()
with open("F_W.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
model.save_weights('F_W.h5')
print('Weights saved')