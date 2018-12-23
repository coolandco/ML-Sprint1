import keras
import os
import skimage

from skimage import io
from skimage.filters import threshold_minimum as threshold # seemed to have good results
import numpy as np


for filename in os.listdir('digitsToPredict'):
    print('=============================================')
    print('Prediction for Image ' + filename + ':')

    # Read Image
    image = io.imread('digitsToPredict/' + filename)

    image = skimage.color.rgb2grey(image)  # For images, that are not greyscale

    # Make image Black/White
    thresh = threshold(image)
    binary_image = image > thresh

    # Print the Image to console
    for line in binary_image:
        for pixel in line:
            if pixel == False:
                print(' ', end='')
            else:
                print('@', end='')
        print('\n', end='')

    #  Because we start a prediction for every file on its own,
    #  we need to expand the shape to (1,28,28,1)
    binary_image = np.expand_dims(binary_image, axis=0)
    #  print(binary_image.shape)
    binary_image = np.expand_dims(binary_image, axis=3)
    #  print(binary_image.shape)


    #  Load the trained model
    model = keras.models.load_model('model.h5')

    #  Start the prediction
    prediction = model.predict_classes(binary_image)
    print('Predicted number of Image ' + str(filename) + ' is ' + str(prediction[0]))



