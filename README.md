# bundestag-classification

The files in this repository can be used to build a classifier on German parliamentary speeches. In the following the steps to execute the code after cloning it locally are listed.

## Installations
I order to avoid package conflicts, it is recommended to create a virtual environment first.
Then run pip install -r requirements.txt
 
## data acquisition 
Run the file scraper.py in the subfolder Files. After that, the speeches as xml files will be in the same subfolder.

## preprocessing
Run the file data-processing.py. The speeches are saved in a pickle file with a dictionary of spacy docs. If you want to get a pickle file with the dictionary of the speeches as strings, just remove the comments in lines 145f

## feature extraction
The methods for this are in features.py, so run this file first. Afterwards run extraction.py. This produces a pickle file feature_matrices.py with the panda dataframe of features for all speech instances.

## classifier
Run classifier.py. This outputs the results to the console. Furthermore, for every tested machine-learning algorithm, a confusion matrix is created and saved as a png. This file runs for about 90 minutes on a normal CPU with 8GB RAM due to hyperparameter optimization.
