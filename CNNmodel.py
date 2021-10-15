# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization#,Activation
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

##
resize = 337
batchsize = 16
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
model.add(Conv2D(filters=16,
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
# pooling layer
model.add(MaxPooling2D(pool_size=(2,2),strides=None))
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

model.add(Dense(64,activation='relu'))

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

train_history = model.fit_generator(
    train_data,
    epochs=10,
    #steps_per_epoch = 80,
    validation_data = valid_data,
    
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

model.save('first_model_10epoch.h5')

""" first_model_10epoch.h5
train_loss = [0.3909986879879027, 0.36521844220945726, 0.33862194580685046, 0.3237281373260624, 0.30690500733076426, 0.2935682059749234, 0.29575925391759483, 0.2736223728346285, 0.27457857864537083, 0.24923820775493769]
train_acc = [0.84877694, 0.8532353, 0.8646825, 0.87335825, 0.87685263, 0.88287747, 0.87853956, 0.8901072, 0.89311963, 0.90336186]
val_loss = [0.05320332199335098, 0.8053330183029175, 0.027755873277783394, 0.6836038827896118, 0.2334701120853424, 0.43213951587677, 0.6126859188079834, 0.3879204988479614, 0.19773878157138824, 0.8002468347549438]
val_acc = [0.8626187443733215, 0.861940324306488, 0.8653324246406555, 0.8127543926239014, 0.855495274066925, 0.7795115113258362, 0.8575305342674255, 0.8653324246406555, 0.8856852054595947, 0.7879918813705444]
score = [0.09579435735940933, 0.900923490524292]
"""

"""
train_loss = [0.3825,0.3225,0.2935]
train_acc = [0.8532,0.8667,0.8824]
val_loss = [0.1491,0.0494,0.3126]
val_acc = [0.8392,0.8365,0.8860]
score = [0.08772, 0.9358]
"""