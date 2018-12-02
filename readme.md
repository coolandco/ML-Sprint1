## Installation Process:
I tried a little bit around to get it working, this is what i came up with:

Notes:
- Wait until the commitment process is finished, bevore pushing. Otherwise corrupted data.

### Prequisites:

- PyCharm (start the 64Bit Version)
- Latest Python 3.6 (64Bit)

### Installation
1. Tensorflow:
    - Upgrade your pip in your venv:
		- Run the following two times, if it fails the first time
        - PyCharm Terminal: `python -m pip install -U --force-reinstall pip`
    - PyCharm Terminal `pip install --upgrade tensorflow`

2. Keras
    - PyCharm Terminal `pip install keras`

3. Skyimage
    - PyCharm Terminal `pip install scikit-image`