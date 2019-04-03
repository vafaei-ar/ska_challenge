##############################################################
####### This code can categorize objects using CNN ###########
##############################################################


from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Input,Reshape,Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import Model
import numpy as np
from keras.utils import np_utils
import random

fraction_train = 70;fraction_test = 15;fraction_dev = 15;

Data = np.float32(np.load('DATA1_SKAMid_B5_1000h_v2.npy')/255.);
Label = np.load('Label_SKAMid_B5_1000h_v2.npy');
catLabel = np_utils.to_categorical(Label)

size_data = Label.shape[0]
random_index=random.sample(range(0, size_data), size_data)

b=np.uint(np.ceil((fraction_test*size_data)/100));
a=np.uint(np.ceil((fraction_train*size_data)/100))

train_index = random_index[0:a]; x_train=Data[train_index];y_train=catLabel[train_index];
test_index = random_index[a:a+b]; x_test=Data[test_index];y_test=catLabel[test_index];
dev_index = random_index[a+b:];x_dev=Data[dev_index];y_dev=catLabel[dev_index];

###################### Input Layer###############################
x = Input((10,10,1))

###################### Conv Layer ################################
layer_1 = Conv2D(32,3,activation='relu',padding='same')(x)
layer_2 = Conv2D(32,3,activation='relu',padding='same')(layer_1)
layer_3 = MaxPooling2D(2)(layer_2)

layer_4 = Conv2D(64,3,activation='relu',padding='same')(layer_3)
layer_5 = Conv2D(64,3,activation='relu',padding='same')(layer_4)
layer_6 = MaxPooling2D(2)(layer_5)

layer_7 = Conv2D(128,3,activation='relu',padding='same')(layer_6)
layer_8 = Conv2D(128,3,activation='relu',padding='same')(layer_7)
layer_9 = MaxPooling2D(2)(layer_8)

###################### FC Layer ##################################
layer_10 = Flatten()(layer_9)
layer_11 = Dense(256,activation='relu')(layer_10)
y = Dense(3,activation='softmax')(layer_11)

###################### Model ####################################
cnnModel = Model(x,y)
cnnModel.compile(optimizer='Adam',loss=categorical_crossentropy,metrics=['accuracy'])
cnnModel.summary()
#from  keras.utils import plot_model
#plot_model(cnnModel,'cnnModel.png',show_shapes=True)
cnnModel.fit(x_train,y_train,epochs=10,batch_size=32,shuffle=True,validation_data=(x_dev,y_dev))

score = cnnModel.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



################################################################
## We had two sets of images, 10x10, 150x150; in the following##
## we are training 150x150 images ##############################


fraction_train = 70;fraction_test = 15;fraction_dev = 15;

Data = np.float32(np.load('DATA2_SKAMid_B5_1000h_v2.npy')/255.);
Label = np.load('Label_SKAMid_B5_1000h_v2.npy');
catLabel = np_utils.to_categorical(Label)

size_data = Label.shape[0]
random_index=random.sample(range(0, size_data), size_data)

b=np.uint(np.ceil((fraction_test*size_data)/100));
a=np.uint(np.ceil((fraction_train*size_data)/100))

train_index = random_index[0:a]; x_train=Data[train_index];y_train=catLabel[train_index];
test_index = random_index[a:a+b]; x_test=Data[test_index];y_test=catLabel[test_index];
dev_index = random_index[a+b:];x_dev=Data[dev_index];y_dev=catLabel[dev_index];

###################### Input Layer###############################
x = Input((150,150,1))

###################### Conv Layer ################################
layer_1 = Conv2D(32,3,activation='relu',padding='same')(x)
layer_2 = Conv2D(32,3,activation='relu',padding='same')(layer_1)
layer_3 = MaxPooling2D(2)(layer_2)

layer_4 = Conv2D(64,3,activation='relu',padding='same')(layer_3)
layer_5 = Conv2D(64,3,activation='relu',padding='same')(layer_4)
layer_6 = MaxPooling2D(2)(layer_5)

layer_7 = Conv2D(128,3,activation='relu',padding='same')(layer_6)
layer_8 = Conv2D(128,3,activation='relu',padding='same')(layer_7)
layer_9 = MaxPooling2D(2)(layer_8)

###################### FC Layer ##################################
layer_10 = Flatten()(layer_9)
layer_11 = Dense(256,activation='relu')(layer_10)
y = Dense(3,activation='softmax')(layer_11)

###################### Model ####################################
cnnModel = Model(x,y)
cnnModel.compile(optimizer='Adam',loss=categorical_crossentropy,metrics=['accuracy'])
cnnModel.summary()
#from  keras.utils import plot_model
#plot_model(cnnModel,'cnnModel.png',show_shapes=True)
cnnModel.fit(x_train,y_train,epochs=10,batch_size=32,shuffle=True,validation_data=(x_dev,y_dev))

score = cnnModel.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])