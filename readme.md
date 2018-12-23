# Running:
Use Python 3 to start following scripts:

- ModelCreation.py
    - Trains a model with the mnist dataset.
    - Places model into current directory
- Prediction.py
    - Takes the model from the current directory
    - Makes a prediction with the images from the 'digitsToPredict' directory
    - digit3.png is the given image, we should use for this exercise

# Installation Process for Development:
I tried a little bit around to get it working, this is what I came up with:

### Prequisites:

- PyCharm (start the 64Bit Version)
- Latest Python 3.6 (64Bit)

### Installation of the imports in PyCharm
1. Tensorflow:
    - Upgrade your pip in your Virtual Enviroment (venv):
		- Run the following command two times, if it fails the first time
        - PyCharm Terminal: `python -m pip install -U --force-reinstall pip`
    - PyCharm Terminal `pip install --upgrade tensorflow`

2. Keras
    - PyCharm Terminal `pip install keras`

3. Skyimage
    - PyCharm Terminal `pip install scikit-image`
