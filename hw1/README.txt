Code for this project is located at 

To run the code, first install the /code/requirements.txt file with pip 19.0 or above. 

Then run the code by running the /code/comparison.py file directly. This script will pull the two datasets and run the classifiers of interest in the assignment; no additional dataset sources are needed, the datasets are pulled in directly from external repositories via the script.

Controls for the script are contained in the “__main__” section. 

Choose which dataset to run by including in either “fashion” , “digits”, or both into the list variable named datasets. 

Cross validation code is written. To run cross validation in addition to the classifiers without cross validation, change the variable called “crossvalidation_toggle” to True from False. 

Note that all classifiers are pulled from the implementation in the sklearn package.

To see which classifiers are being run, see the function create_models(). Comment out which classifiers you may not be interested in running. All accuracy results will be outputted as .csv files to the folder of the repo in which the code is run. 