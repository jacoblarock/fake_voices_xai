# fake_voices_xai

This repository is a setup for experiments using explainable models to classify deepfake/bona-fide
audio samples.  
It consists of feature extraction functions and data preparation to create usable inputs for models
as well as a creation, training and evaluation setup for a traditional (not explainable) model, with
the intent of using it as a surrogate for a later explainable model. Explanations are generated
using LIME (details below).

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

In this repository, there are several feature extractors that are used to translate audio samples
into useful input for classification models.  
IF a feature extractor is based on a feature which is novel and specific to certain research or from
an article, it is cited in the comments of the code (references below). Standard features that are
typical in detection models and not specific to certain research or articles, or features that are
built in to existing libraries (for example MFCCs) are not cited.
  
The features used in this repository can be, as in my previous research, classified into perceptible
and imperceptible features. Perceptible features are features that are able to be perceived by
humans, such as vocal qualities, for example timbre, jitter or pitch fluctuations. Such features are
often the reason that fake audio samples often sound somewhat "uncanny" to the ear. Imperceptible
features, however, are features that are typically not able to be perceived by humans. These include
spectrographic features, such as mel-spectrograms, and spectral coefficients, such as mel-frequency
cepstral coefficients or linear frequency cepstral coefficients. Previous research has often
concluded that the "problems" in fake audio samples identified using such features are often in very
low or high frequency ranges, which can be classified as "speaker independent features" [[1]](#1).

# Datasets
The In-The-Wild Dataset has been primarily used for experiments because of its relative recency and
focus on generalization. It can be found here [[6]](#6):
```
https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download
```

Alternatively, the dataset of ASVSpoof2021 DF can be found here [[7]](#7):
```
https://zenodo.org/records/4835108
```
Datasets can be put in the `datasets` directory, which is in the .gitignore file.

# Models
There are pickle dumps of trained models in the trained_models directory of this repository that can
be used for evaluation or surrogate purposes. The file names are a summary of the features involved
and the training. Summaries of the models as well as a graphic of their architectures are given
the readme files in their respective directories (in progress).  

### Example: Architecture of the featured model in the repository
![architecture of the featured model, others in the trained_models directory](model_plot.png)

### Why separated sub-models?
As can be seen in the diagram of the model, each input feature is first processed in its own
sub-model before being concatenated and run through two final layers, landing at a singular output
terminus. Because several of the features have a different input shape, it is not practical to
concatenate the features together for use as an input for a combined model without producing a large
amount of redundant data and potentially misrepresenting the features in their original form. code
for pre- model concatenation is present in the repository under the "lines" training method in the
classification file, but is presently not used. I have not deleted it in case it may become
practical in the future.

# Explanations
As of now, the explanations are implemented using local interpretable model-agnostic explanations
(LIME). The implementation functions by generating intermediate evaluation data, the outputs of
each of the sub-models before the concatenate layer, from the initial training data to use as a 
reference for the local explainer to create its linear approximation. For the creation of an
explanation for the prediction on a new sample, the intermediate prediction data will be generated
in the same fashion and used as the datapoint to explain.

### Why use intermediate data and not the original inputs?
The intermediate data serves the purpose of offering a direct way to associate the input features
with the end result, while maintaining a uniform input shape for the explainer, as the built-in
tabular explainer library in LIME does not support non-uniform input shapes.  
Drawing an association between the output of the terminus and the outputs of the submodels for each
feature allows to assess the importance of each individual input feature, as the submodels have no
interaction with one-another. In short, the total importance of the output of the submodels will be
equal to the importance of the inputs of each of the submodels.

# Notes on Implementation

### Feature Extractors
For these experiments various feature are applied to given datasets, resulting in dataframes
containing arrays of extracted features. These can be, for each sample in the dataset, ints,
one-dimensional arrays or two-dimensional arrays.  
For increased efficiency, there is a generic implementation of multithreading for file-based feature
extraction (in mt_operations).

### Data Preparation
In order to prepare data for input into neural networks, a standardized array shape is needed.
Because the extracted features do not result in standardized array shapes a sliding window is passed
over every array for standardization.  

### Networks
Each feature is processed separately by its own network. These networks are stitched together into a
kind of meta-network to produce the end-classification.

### Training and Evaluation
Training is performed on batches of previously extracted features merged together and labelled.  
Evaluation is performed akin to training on batched of labelled features. Because of the nature of 
the sliding window over every file, there will be multiple evaluation results for every sample.
These results are averaged and filtered by a threshold to produce a final result.

### Note
I have tried my best to include pydoc where it is relevant.

# References
<a id="1">[1]</a>
Xin Liu et al.  
“Hidden-in-Wave: A Novel Idea to Camouflage AI-Synthesized Voices Based on Speaker-Irrelative
Features”. In: 2023 IEEE 34th International Symposium on Software Reliability Engineering (IS-SRE).
2023 IEEE 34th International Symposium on Software Reliability Engineering (ISSRE). Florence, Italy:
IEEE, Oct. 9, 2023, pp. 786–794. isbn: 9798350315943. doi: 10 . 1109 / ISSRE59848 . 2023 . 00029.
url: https : / / ieeexplore . ieee . org / document / 10301243/ (visited on 05/28/2024).  
<a id="2">[2]</a>
Anuwat Chaiwongyen et al.  
“Deepfake-speech Detection with Pathological Features and Multilayer Perceptron Neural Network”. In:
2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference
(APSIPA ASC). 2023 Asia Pacific Signal and Information Processing Association Annual Summit and
Conference (APSIPA ASC). Taipei, Taiwan: IEEE, Oct. 31, 2023, pp. 2182–2188. isbn: 9798350300673.
doi: 10 . 1109 / APSIPAASC58517 . 2023 . 10317331. url:
https://ieeexplore.ieee.org/document/10317331/ (visited on 05/28/2024).  
<a id="3">[3]</a>
Anuwat Chaiwongyen et al.  
“Contribution of Timbre and Shimmer Features to Deepfake Speech Detection”. In: 2022 Asia-Pacific
Signal and Information Processing Association Annual Summit and Conference (AP-SIPA ASC). 2022 Asia
Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC).
Chiang Mai, Thailand: IEEE, Nov. 7, 2022, pp. 97–103. isbn: 978-616-590-477-3. doi:
10.23919/APSIPAASC55919.2022.9980281. url: https://ieeexplore.ieee.org/document/9980281/ (visited
on 05/28/2024).  
<a id="4">[4]</a>
Menglu Li, Yasaman Ahmadiadli, and Xiao-Ping Zhang.
“A Comparative Study on Physical and Perceptual Features for Deepfake Audio Detection”. In:
Proceedings of the 1st International Workshop on Deepfake Detection for Audio Multimedia. MM ’22:
The 30th ACM International Conference on Multimedia. Lisboa Portugal: ACM, Oct. 14, 2022, pp. 35–41.
isbn: 978-1-4503-9496-3. doi: 10.1145/3552466.3556523. url:
https://dl.acm.org/doi/10.1145/3552466.3556523 (visited on 05/11/2024).  
<a id="5">[5]</a>
Zahra Khanjani et al.
“Learning to Listen and Listening to Learn: Spoofed Audio Detection Through Linguistic Data
Augmentation”. In: 2023 IEEE International Conference on Intelligence and Security Informatics
(ISI). 2023 IEEE International Conference on Intelligence and Security Informatics (ISI). Charlotte,
NC, USA: IEEE, Oct. 2, 2023, pp. 01–06. isbn: 9798350337730. doi: 10.1109/ISI58743.2023.10297267.
url: https://ieeexplore.ieee.org/document/10297267/ (visited on 05/28/2024).  
<a id="6">[6]</a>
Nicolas M. Müller et al.
Does Audio Deepfake Detection Generalize? Apr. 21, 2022. arXiv: 2203.16263[cs,eess]. url:
http://arxiv.org/abs/2203.16263 (visited on 07/04/2024).  
<a id="7">[7]</a>
Junichi Yamagishi et al.
“ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection”. In: 2021 Edition of
the Automatic Speaker Verification and Spoofing Countermeasures Challenge. 2021 Edition of the
Automatic Speaker Verification and Spoofing Countermeasures Challenge. ISCA, Sept. 16, 2021, pp.
47–54. doi: 10 . 21437 / ASVSPOOF . 2021 - 8. url:
https://www.isca-archive.org/asvspoof_2021/yamagishi21_asvspoof.html (visited on 07/04/2024).  
