## Installation Process:
I tried a little bit around to get it working, this is what i came up with:

### Prequisites:

- PyCharm (start the 64Bit Version)
- Latest Python 3.6 (64Bit)

### Installation
... is done in PyCharm:
1. Tensorflow:
    - Upgrade your pip in your Virtual Enviroment (venv):
		- Run the following command two times, if it fails the first time
        - PyCharm Terminal: `python -m pip install -U --force-reinstall pip`
    - PyCharm Terminal `pip install --upgrade tensorflow`

2. Keras
    - PyCharm Terminal `pip install keras`

3. Skyimage
    - PyCharm Terminal `pip install scikit-image`

# Running:

- ModelCreation.py
    - Trains a model for the mnist dataset.
    - Places model into current directory
- Prediction.py
    - Takes the model from the current directory
    - Makes a prediction with the images from the 'digitsToPredict' directory