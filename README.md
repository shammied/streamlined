# STREAMLInED
Shared Tasks for Rapid, Efficient Analysis of Many Languages in Emerging Documentation.

Shared Task for Spoken Language Identification of Endanged Languages

University of Washington research project

contact: debnaths [at] uw [dot] edu

## Data

Data is collected from [The Endangered Language Archive (ELAR)](https://www.elararchive.org/). The table below describes some basic information about the languages used and the number of hours worth of data.

| **Language** | **Family**   | **Location**     | **Hours** |
|--------------|--------------|------------------|-----------|
| Pech         | Chibchan     | Honduras         | 1.26      |
| Caquinte     | Arawaken     | Peru             | 2.10      |
| Kwa          | Niger-Congo  | Nigeria          | 2.31      |
| Lakumurau    | Austronesian | Papua New Guinea | 2.03      |
| N\|u         | Tuu          | South Africa     | 1.28      |
| Saaroa       | Austronesian | Taiwan           | 1.13      |
| Irantxe      | -            | Brazil           | 2.41      |
| Puma         | Sino-Tibetan | Nepal            | 2.10      |

`src/data_preprocessing.py`: trim and manipulate volume of audio file

`src/feature_extraction.py`: extract MFCC as features from audio files

## Baseline Systems

### Random Forest Classifier

`src/langid.py` and `src/rf.py` : train random forest or svm model with the parameters below:

```python
n_estimators = 100
max_depth = None
max_features = "sqrt"
min_samples_leaf = 1
min_samples_split = 2
bootstrap = True
criterion = "gini"
```

### Feed Forward Neural Network

`src/ffnn.py `: train feed forward neural network model with the hyperparameters below:

```python
input_dim = 1*20
hidden_dim = 100
output_dim = 8
batch_size = 1024
learning_rate = 0.01
num_epochs = 400
```