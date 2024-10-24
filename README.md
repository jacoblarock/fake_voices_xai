# fake_voices_xai

Setup for experiments using explainable models to classify deepfake/bona-fide audio samples

# Setup

In order to set-up the environment, use the provided anaconda environment contained within the environment.yml file:
```
conda env create -r environment.yml
```
Then, activate the environment using:
```
conda activate fake_voices
```

# Feature Extractors

In this repo are several feature extractors that are used to translate audio samples into useful input for classification models. 
Included are both perceptible and imperceptible features based on several sources of previous research.

# test.py

The `test.py` file is used for testing, and its contents are not especially relevant.
