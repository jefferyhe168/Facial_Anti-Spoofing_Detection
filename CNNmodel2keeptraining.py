# -*- coding: utf-8 -*-

from keras import models
#from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization#,Activation
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
    target_size=(resize, resize), # picture change to resize x resize
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
# load model
model = models.load_model('model/second_model_8+7+4epoch.h5')
# load model weight
model.load_weights('model/second_model_weight_2/weights-improvement-04-0.89.hdf5')

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
    epochs=4,
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

model.save('second_model_remaining2epoch.h5')

def show_accuracy_history(train,validation):
    plt.plot(train)
    plt.plot(validation)
    plt.title('Train History')
    plt.ylabel('accuracy') #accuracy
    plt.xlabel("Epoch")
    plt.legend(['train','validation'],loc='upper right')
    plt.show()
def show_loss_history(train,validation):
    plt.plot(train)
    plt.plot(validation)
    plt.title('Train History')
    plt.ylabel('loss') #loss
    plt.xlabel("Epoch")
    plt.legend(['train','validation'],loc='upper right')
    plt.show()

# the previous training data(0-8epoch)
train_acc = [0.8444391, 0.8540788, 0.86420053, 0.8678154, 0.87914205, 0.8842029, 0.8889023, 0.8889023]
train_loss = [0.5343604628717142, 0.3563621298909719, 0.34680493119424016, 0.3284489999698406, 0.3061284246193614, 0.28763556225441467, 0.2737413722968328, 0.2693996011442356]
val_acc = [0.8626187443733215, 0.8578697443008423, 0.8490502238273621, 0.8612618446350098, 0.869742214679718, 0.8107191324234009, 0.8385345935821533, 0.882632315158844]
val_loss = [0.18820221722126007, 0.01958003267645836, 0.4764973521232605, 0.2846358120441437, 0.3661839962005615, 0.23174956440925598, 0.2641424834728241, 0.12734660506248474]
scores = [0.20904909074306488, 0.8824537992477417]
# merge the new and the previous data
train_acc.extend(train_history.history['accuracy'])
train_loss.extend(train_history.history['loss'])
val_acc.extend(train_history.history['val_accuracy'])
val_loss.extend(train_history.history['val_loss'])
# plot the total training data
show_accuracy_history(train_acc, val_acc)
show_loss_history(train_loss, val_loss)

# some experiment data
"""
# remaining 4 epoch
train_acc = [0.9009519, 0.90203637, 0.9181829, 0.91637546]
train_loss = [0.25437171435640554, 0.2567818648614917, 0.2130645158754399, 0.2148724562171023]
val_acc = [0.8575305342674255, 0.8130936026573181, 0.8616011142730713, 0.8860244154930115]
val_loss = [0.15843766927719116, 0.5664968490600586, 0.7407354116439819, 0.10155582427978516]
scores = [0.20904909074306488, 0.8824537992477417]
"""
"""
# remaining 2 epoch(for optimizer adagrad testing)
train_acc = [0.8714303, 0.8911917]
train_loss = [0.5285601950234743, 0.27804099166805535]
val_acc = [0.7751017808914185, 0.755088210105896]
val_loss = [0.5716615915298462, 1.2871556282043457]
scores = [0.20904909074306488, 0.8824537992477417]
"""
"""
# 8 + remaining 7 epoch(for optimizer adagrad testing)
train_acc = [0.8786601, 0.89878297, 0.90914565, 0.9103506, 0.9220388, 0.9260152, 0.9283046]
train_loss = [0.3646299697818749, 0.25655201447700965, 0.22980236910541224, 0.21897662155835626, 0.20432443717044152, 0.19188979187993088, 0.1823107072898804]
val_acc = [0.8690637946128845, 0.8130936026573181, 0.7737449407577515, 0.8297150731086731, 0.8042740821838379, 0.770691990852356, 0.7540705800056458]
val_loss = [0.48254936933517456, 0.7057442665100098, 0.32581961154937744, 0.3206760883331299, 0.13369707763195038, 0.7547418475151062, 0.3941769599914551]
scores = [0.20904909074306488, 0.8824537992477417]
"""
"""
# first 10 epoch
train_acc = [0.8444391, 0.8540788, 0.86420053, 0.8678154, 0.87914205, 0.8842029, 0.8889023, 0.8889023, 0.90336186, 0.90757924]
train_loss = [0.5343604628717142, 0.3563621298909719, 0.34680493119424016, 0.3284489999698406, 0.3061284246193614, 0.28763556225441467, 0.2737413722968328, 0.2693996011442356, 0.24509562268370189, 0.245634181343054]
val_acc = [0.8626187443733215, 0.8578697443008423, 0.8490502238273621, 0.8612618446350098, 0.869742214679718, 0.8107191324234009, 0.8385345935821533, 0.882632315158844, 0.6556987762451172, 0.8229308128356934]
val_loss = [0.18820221722126007, 0.01958003267645836, 0.4764973521232605, 0.2846358120441437, 0.3661839962005615, 0.23174956440925598, 0.2641424834728241, 0.12734660506248474, 0.1303090751171112, 0.4465087652206421]
scores = [0.20904909074306488, 0.8824537992477417]
"""