Nutrition-Predictor
===================

Using a python Virtual Environment (https://pypi.python.org/pypi/virtualenv) is a great way to manage python packages.

After cloning for the first time:
- pip install virtualenv # Install the virtualenv package manager
- virtualenv venv # Create the directory that will hold the python packages
- pip install -r requirements.txt # Install the required python packages

Each time you start using it:
- . venv/bin/activate # Start using the virtual environment

If you need to add a new python package:
- pip install NEW_PACKAGE # Make sure you have already activated the virtual environment
- pip freeze > requirements.txt # Add the package to the list of required packages
