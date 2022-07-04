#imports from arl-eegmodels readme
#from EEGModels import EEGNet
from tensorflow.keras.models import Model
#from deepexplain.tensorflow import DeepExplain
from tensorflow.keras import backend as K

#imports from eegnetfrom tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

#sentdex imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
import os
import random
import time

#This compares trains a model to which is saved in the new_models and is what we run testing against.
#####Start of sentdex's data compilation#####################
ACTIONS = ["left", "right", "none"]
#reshape = (-1, 16, 128, 1)
reshape = (-1, 16, 60)

def create_data(starting_dir="D:\\Projects\\ProjectCrypt\\BCI-master\\data_V3\\data"):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            #print(action, item)
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
        training_data[action] = training_data[action][:min(lengths)]

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)
    # creating X, y
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:

            if action == "left":
                combined_data.append([data, [1, 0, 0]])

            elif action == "right":
                #np.append(combined_data, np.array([data, [1, 0]]))
                combined_data.append([data, [0, 0, 1]])

            elif action == "none":
                combined_data.append([data, [0, 1, 0]])

    np.random.shuffle(combined_data)
    print("length:",len(combined_data))
    return combined_data


print("creating training data")
traindata = create_data(starting_dir="D:\\Projects\\ProjectCrypt\\BCI-master\\data_V3\\data")
train_X = []
train_y = []
for X, y in traindata:
    train_X.append(X)
    train_y.append(y)


print("creating testing data")
testdata = create_data(starting_dir="D:\\Projects\\ProjectCrypt\\BCI-master\\data_V3\\validation_data")
test_X = []
test_y = []
for X, y in testdata:
    test_X.append(X)
    test_y.append(y)

print(len(train_X))
print(len(test_X))


print(np.array(train_X).shape)
train_X = np.array(train_X).reshape(reshape)
test_X = np.array(test_X).reshape(reshape)

train_y = np.array(train_y)
test_y = np.array(test_y)


#Start of EEGNet########################################################################
def EEGNet(nb_classes = 3, Samples = 128, Chans = 16,  #64 by default ask nao. #make nume
#What is nb_classes? I think 3 for directions (like sentdexes), Yes ERP uses 4 for ears and visuals. Nao uses 4 for directions.
             dropoutRate = 0.5, kernLength = 64, F1 = 8,
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
##ERP uses
#model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples,
#               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16,
#               dropoutType = 'Dropout')


        if dropoutType == 'SpatialDropout2D': #ccompare differences between dropouts
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        input1   = Input(shape = (Chans, Samples, 1))

        ##################################################################
        block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                       input_shape = (Chans, Samples, 1),
                                       use_bias = False)(input1)
        block1       = BatchNormalization()(block1)
        block1       = DepthwiseConv2D((Chans, 1), use_bias = False,
                                       depth_multiplier = D,
                                       depthwise_constraint = max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)
        block1       = Activation('elu')(block1)
        block1       = AveragePooling2D((1, 4))(block1)
        block1       = dropoutType(dropoutRate)(block1)

        block2       = SeparableConv2D(F2, (1, 16),
                                       use_bias = False, padding = 'same')(block1)
        block2       = BatchNormalization()(block2)
        block2       = Activation('elu')(block2)
        block2       = AveragePooling2D((1, 8))(block2)
        block2       = dropoutType(dropoutRate)(block2)

        flatten      = Flatten(name = 'flatten')(block2)

        dense        = Dense(nb_classes, name = 'dense',
                             kernel_constraint = max_norm(norm_rate))(flatten)
        softmax      = Activation('softmax', name = 'softmax')(dense)

        return Model(inputs=input1, outputs=softmax)
#end of eegnet####################################################
model = EEGNet(nb_classes = 3, Samples = 128, Chans = 16,
             dropoutRate = 0.5, kernLength = 64, F1 = 8,
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              #also try with removing metrics

#fitting the model from from sentdex#################
epochs = 10  #one complete pass of the training dataset through the algorithm
batch_size = 32
for epoch in range(epochs):
    model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(test_X, test_y))
    score = model.evaluate(test_X, test_y, batch_size=batch_size)
    #print(score)

#ERP uses the below to fit their model
#fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300,
                        #verbose = 2, validation_data=(X_validate, Y_validate),
                        #callbacks=[checkpointer], class_weight = class_weights)

#Nao uses
#epochs = 200
#model = EEGNet()
#model = model.to(device)
#model_path = 'model.pth'
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.model"
model.save(MODEL_NAME)
print("saved:")
print(MODEL_NAME)
