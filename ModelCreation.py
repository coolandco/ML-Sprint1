from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K

num_classes = 10

################################################
# Preprocessing. Leave this section unchanged! #
################################################

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()


##################
# Implement here #
##################

#
#  All properties of the layers, that are commented, are from the E-Mail
#  All other properties are assumptions, that were tested and seemed to be the right choice
#


# add to the model: Convolution layer with 32 output filters, a kernel size of 3x3
model.add(keras.layers.Conv2D(
    filters=32,  # 32 output filters
    kernel_size=(3, 3),  # kernel size of 3x3
    input_shape=(28, 28, 1),  # for 28x28 pictures with one channel
    data_format='channels_last'  # The channel ist last in the data
    )
)

# Convolution layer with 64 output filters, a kernel size of 3x3
model.add(keras.layers.Conv2D(
    filters=64,  # 64 output filters
    kernel_size=(3, 3),  # kernel size of 3x3
    data_format='channels_last'  # The channel ist last in the data
    )
)

# Maxpooling layer with a pool size of 2x2
model.add(keras.layers.MaxPooling2D(
    pool_size=(2, 2),  # pool size of 2x2
    strides=None,
    padding='valid',
    data_format='channels_last'
    )
)

# Dropout layer with a drop fraction of 0.5
model.add(keras.layers.Dropout(
    rate=0.5,  # drop fraction of 0.5
    noise_shape=None,
    seed=None
    )
)

# Flatten layer
model.add(keras.layers.Flatten(
    data_format='channels_last'
    )
)

# Fully-connected layer with 128 neurons
model.add(Dense( # Fully-connected layer
        128, # with 128 neurons
        activation='relu'
    )
)

# Dropout layer with a drop fraction of 0.5
model.add(keras.layers.Dropout(
    rate=0.5,  # drop fraction of 0.5
    noise_shape=None,
    seed=None
    )
)

# Fully-connected layer with as many neurons as there are classes in the problem (Output layer), activation function: Softmax
model.add(Dense(
    10, #  10 possible Digits -- 10 Classes
    activation='softmax'
    )
)

print('Configuration of model: ', model.get_config())


adam = keras.optimizers.Adam(lr=0.001,  # Learning rate: 0.001
                             beta_1=0.9,
                             beta_2=0.999,
                             epsilon=None,
                             decay=0.0,
                             amsgrad=False
                             )

model.compile(optimizer=adam,  # Optimizer: Adam
              loss="categorical_crossentropy",  # Loss: Categorical Crossentropy
              metrics=['accuracy']  # Evaluation Metric: Accuracy
              )

model.fit(x_train, y_train,
          epochs=3,  # Epochs: 3
          batch_size=128  # Batch size: 128
          )

score = model.evaluate(x_test, y_test, batch_size=128)

print('Evaluation score:  ', score)

model.save("model.h5") # saves / overwrites the model