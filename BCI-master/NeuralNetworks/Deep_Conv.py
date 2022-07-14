"""
===>Taken from arl-eegmodels-master/README.md

#To use this package, place the contents of this folder in your PYTHONPATH environment variable. Then, one can simply import any model and configure it as
#refer to ARL-eegmodels-master to find out what else to add
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from EEGModels import EEGNet
from tensorflow.keras.models import Model
from deepexplain.tensorflow import DeepExplain
from tensorflow.keras import backend as K

model3 = DeepConvNet(nb_classes = ..., Chans = ..., Samples = ...)

#Compile the model with the associated loss function and optimizer (in our case, the categorical cross-entropy and Adam optimizer, respectively).
#Then fit the model and predict on new test data.

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')   #or model.compile(loss = ..., optimizer = ..., metrics = ...)
fittedModel    = model.fit(...)
predicted      = model.predict(...)


# configure, compile and fit the model
"""

""" ==>taken from arl-eegmodels-master/EEGModels.py

Keras implementation of the Deep Convolutional Network as described in
Schirrmeister et. al. (2017), Human Brain Mapping.

Citation:
    @article{hbm23730,
    author = {Schirrmeister Robin Tibor and
              Springenberg Jost Tobias and
              Fiederer Lukas Dominique Josef and
              Glasstetter Martin and
              Eggensperger Katharina and
              Tangermann Michael and
              Hutter Frank and
              Burgard Wolfram and
              Ball Tonio},
    title = {Deep learning with convolutional neural networks for EEG decoding and visualization},
    journal = {Human Brain Mapping},
    volume = {38},
    number = {11},
    pages = {5391-5420},
    keywords = {electroencephalography, EEG analysis, machine learning, end‐to‐end learning, brain–machine interface, brain–computer interface, model interpretability, brain mapping},
    doi = {10.1002/hbm.23730},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730}
    }

This implementation assumes the input is a 2-second EEG signal sampled at
128Hz, as opposed to signals sampled at 250Hz as described in the original
paper. We also perform temporal convolutions of length (1, 5) as opposed
to (1, 10) due to this sampling rate difference.

Note that we use the max_norm constraint on all convolutional layers, as
well as the classification layer. We also change the defaults for the
BatchNormalization layer. We used this based on a personal communication
with the original authors.

                  ours        original paper
pool_size        1, 2        1, 3
strides          1, 2        1, 3
conv filters     1, 5        1, 10

Note that this implementation has not been verified by the original
authors.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, (1, 5),
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)

    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)

    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)

    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)

    flatten      = Flatten()(block4)

    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
