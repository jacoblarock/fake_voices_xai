# fake_voices_xai

This repository is a setup for experiments using explainable models to classify deepfake/bona-fide audio samples.  
It consists of feature extraction functions and data preparation to create usable inputs for models as well as
a creation, training and evaluation setup for a traditional (not explainable) model, with the intent of using it as a surrogate for a later explainable model.
Implementation of explainability using LIME or similar is planned in the near future.

# Setup

In order to set-up the environment, use the provided anaconda environment contained within the 
environment.yml file:
```
conda env create -f environment.yml
```
Then, activate the environment using:
```
conda activate fake_voices
```

# Feature Extractors

In this repository, there are several feature extractors that are used to translate audio samples into useful 
input for classification models.  
Included are both perceptible and imperceptible features based on several sources of previous.  
IF a feature extractor is based on a feature found in the research, it is cited. Standard features (for example MFCCs) are not cited
research.

# Datasets
To download the In-The-Wild Dataset:
```
https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download
```

The dataset of ASVSpoof2021 DF can be found here:
```
https://zenodo.org/records/4835108
```
Datasets can be put in the `datasets` directory, which is in the .gitignore file.

# Models (traditional)
There are pickle dumps of trained models in the trained_models directory of this repository that can be used for evaluation or surrogate purposes. The file names are a summary of the features involved and the training.
- ItW_hnrs_melspec_mfcc_f0len: This model was trained with the following features on the entire In-The-Wild dataset:
	* Harmonic-noise ratios
	* Mel-spectrograms
	* MFCCs
	* Fundamental frequency lengths
- ItW_hnrs_melspec_mfcc_f0len_onsets_intensity_pitch_u9400: This model was trained on the first 9400 samples of the In-The-Wild dataset and evaluated on the rest. In evaluation, the model achieves an accuracy of 90,93% with a threshold of 0.5. It was trained on the following features:
	- Perceptible:
		* Harmonic-noise ratios
		* Fundamental frequency lengths
		* Onset strengths
		* Intensity
		* Pitch fluctuations
	- Imperceptible:
		* Mel-spectrograms
		* MFCCs
- ItW_multi_percep_until10000: This model was trained on the first 10000 samples of the In-The-Wild dataset and evaluated on the rest. In evaluation, the model achieves an accuracy of approximately 94% with a threshold of 0.5. It was trained using the following features:
	- Perceptible:
		* Harmonic-noise ratios
		* Fundamental frequency lengths
		* Onset strengths
		* Intensity
		* Pitch fluctuations
		* Local jitter
		* Rap jitter
		* ppq5 jitter
		* ppq55 jitter
		* Local shimmer
		* Rap shimmer
		* ppq5 shimmer
		* ppq55 shimmer
	- Imperceptible:
		* Mel-spectrograms
		* MFCCs

# Notes on Implementation

### Feature Extractors
For these experiments various feature are applied to given datasets, resulting in dataframes containing arrays of extracted features. These can be, for each sample in the dataset, ints, one-dimensional arrays or two-dimensional arrays.  
For increased efficiency, there is a generic implementation of multithreading for file-based feature extraction (in mt_operations).

### Data Preparation
In order to prepare data for input into neural networks, a standerdizes array shape is needed. Because the extraced features do not result in standardized array shapes a sliding window is passed over every array for standardization.  

### Networks
Each feature is processed separately by its own network. These networks are stitched together into a kind of meta-network to produce the end-classification.

### Training and Evaluation
Training is performed on batches of previously extracted features merged together and labelled.  
Evaluation is performed akin to training on batched of labelled features. Because of the nature of the sliding window over every file, there will be multiple evaluation results for every sample. These results are averaged and filtered by a threshold to produce a final result.

### Note
I have tried my best to include pydoc where it is relevant.
