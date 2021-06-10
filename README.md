# Etude-de-cas--OCR

# set Flask server name as a global variable
export FLASK_APP=server_for_ionic.py
export FLASK_ENV=development

# python dependencies installation
pip3 install Flask jsonpickle opencv-python glob2 pytesseract flask_cors

# document scanner dependencies
pip3 install --upgrade imutils
python3 -m pip install -U pip
python3 -m pip install -U scikit-image

# deep learning
pip3 install --upgrade keras
pip3 install --upgrade tensorflow

# run Flask server
flask run
