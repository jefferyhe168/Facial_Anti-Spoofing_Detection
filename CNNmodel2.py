# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization#,Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

resize = 337
batchsize = 32
# training data 
# construct the training image generator for data augmentation
train_gen = ImageDataGenerator(  #deal with picture(data argmentation)
    zca_whitening=False,
    rotation_range=40,# Degree range for random rotations
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0,
)
train_data = train_gen.flow_from_directory(
    "PATH\\LCC_FASD\\LCC_FASD_training\\",# 目標目錄
    target_size=(resize, resize), # 所有影像調整為 resize x resize
    color_mode="rgb",
    #color_mode="grayscale",
    #class_mode="categorical",
    class_mode="binary",
    batch_size = batchsize,
    shuffle=True, # 是否隨機播放數據
    save_format="png",
    follow_links=False,
    subset=None,
    interpolation="nearest",
)
# validation data
valid_gen = ImageDataGenerator(  #deal with picture(data argmentation)
    zca_whitening=False,
    rotation_range=40,# Degree range for random rotations
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0,
)
valid_data = train_gen.flow_from_directory(
    "PATH\\LCC_FASD\\LCC_FASD_development\\",# 目標目錄
    target_size=(resize, resize), # 所有影像調整為 resize x resize
    color_mode="rgb",
    class_mode="binary",
    batch_size = batchsize,
    shuffle=True, # 是否隨機播放數據
    save_format="png",
    follow_links=False,
    subset=None,
    interpolation="nearest",
)
# test data
test_gen = ImageDataGenerator()
test_data = test_gen.flow_from_directory(
    "PATH\\LCC_FASD\\LCC_FASD_evaluation\\",# 目標目錄
    target_size=(resize, resize), # 所有影像調整為 resize x resize
    color_mode="rgb",
    #color_mode="grayscale",
    #class_mode="categorical",
    class_mode="binary",
    batch_size = batchsize,
    shuffle=True, # 是否隨機播放數據
    save_format="png",
    follow_links=False,
    subset=None,
    interpolation="nearest",
)

model = Sequential()
# convolutional layer
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 strides=(1,1),
                 input_shape=(resize, resize, 3),
                 padding='same',
                 activation='relu',
                 ))

model.add(BatchNormalization())
# pooling layer
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
# convolutional layer
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 ))
# dropout layer
model.add(Dropout(0.2))
# convolutional layer
model.add(Conv2D(32,(3,3),
                 strides=(1,1),
                 padding='same',
                 activation='relu',
                 ))
# pooling layer
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
# flatten layer (2D -> 1D)
model.add(Flatten())
# dropout layer
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))
# dropout layer
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
# dropout layer
model.add(Dropout(0.2))
# out put layer
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              #loss='categorical_crossentropy',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True,
mode='max')
callbacks_list = [checkpoint]

train_history = model.fit_generator(
    train_data,
    epochs=10,
    #steps_per_epoch = 80,
    validation_data = valid_data,
    callbacks=callbacks_list,
    
#     validation_dat nerator(dataFrameTest,expectedFrameTest,batch_size*2),
#     validation_steps = dataFrame.shape[0]/batch_size*2
)
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(['train','validation'],loc='upper right')
    plt.show()

show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(test_data)
print('accuracy=',scores[1])

model.save('second_model_10epoch.h5')

train_acc = train_history.history['accuracy']
train_loss = train_history.history['loss']
val_acc = train_history.history['val_accuracy']
val_loss = train_history.history['val_loss']