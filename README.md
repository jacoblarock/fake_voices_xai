# fake_voices_xai

Setup for experiments using explainable models to classify deepfake/bona-fide audio samples

# Setup

In order to set-up the environment, use the provided anaconda environment contained within the environment.yml file:
```
conda env create -f environment.yml
```
Then, activate the environment using:
```
conda activate fake_voices
```

# Feature Extractors

In this repo are several feature extractors that are used to translate audio samples into useful input for classification models. 
Included are both perceptible and imperceptible features based on several sources of previous research.

# Datasets
To download the In-The-Wild Dataset:
```
https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download
```
The data can then be extracted into the release_in_the_wild/ directory. This directory is in the .gitignore file so that the dataset will not be committed to the repository.

# extracted_features/
In this directory, the python scripts can cache extracted features, so that they can be reused without repeating computations. This directors is also excluded in the .gitignore file.

# test.py

The `test.py` file is used for testing, and its contents are not especially relevant.
