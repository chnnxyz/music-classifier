# Music Genre Classifier

## Dataset

Data provided in the datasets folder consists of equal duration audio files analyzed with [libROSA](https://librosa.github.io/librosa/) for 1000 audio files of 10 different genres: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae and Rock.

Features:

* filename: Processed filename, all filenames have a <genre>.<id>.au structure
* tempo: Beats per minute on the track
* beats: Number of beats processed in the track
* chroma_stft: Chromatic variable, associated to the intensity and frequency of each note
* rmse: Root Mean Square Energy, related to waveform amplitude
* spectral_centroid: mean value for track's energy level
* spectral_bandwidth: absolute value of the interval size for energy
* rolloff: 85% percentile of energy values
* zero_crossing_rate: times the signal passes from negative to positive
* mfcc1-20: Mel Frequency Ceptral Coefficients [read more.](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
  
## Feature cleanup

In order to run a single-label multiclass model, the label was extracted from the data using Pandas.

```python
data = pd.read_csv('datasets\\data.csv')
data['genre'] = data['filename'].str.split('.').str[0]
```

Highly correllated variables were visualized and removed, a major cluster appeared with:

* spectral_centroid
* spectral_bandwidth
* rolloff

Since all three are energy related, and rolloff is the only one describing dispersion, rolloff was kept and the rest was removed. Since all tracks had the same duration (in seconds) tracks with higher tempo recorded more beats. While tempo is very different in reggae and disco, beats are not explanatory enough and are not functional for future model scalability.

```python
X = data.drop(['filename','genre'], axis = 1)
y = data['genre']
X.drop('beats', axis = 1, inplace = True)
X.drop(['spectral_centroid','spectral_bandwidth'], axis = 1, inplace = True)
```

## Models

All models used a 5-fold shuffled train-test split (i.e. 80% train, 20% test with cv). The most important metric is accuracy, since precision or recall won't have heavy impact on production.

### Random Forest (no preprocessing)

* Accuracy: 68%
* F1 scores over 0.8: Classical (0.94), Pop (0.83)

### Random Forest (standard scaled features)

* Accuracy: 70%
* F1 scores over 0.8: Classical (0.91), Pop (0.83), Blues (0.81)

#### Random Forest Variable importance

RMSE and Rolloff were high importance variables, which makes sense since classical music used to not have compressed signals and therefore should have a lower RMSE and rolloff. 

Chroma_stft should be more varied in classical music, but further data is needed to analyze

### Tensorflow (FFNN: i-Dense(128)-Dropout(.25)-Dense(784)-Dropout(.25)-Dense(784)-Dropout(.25)-Dense(10,softmax), Adam)

* Accuracy: 71%
* F1 scores over 0.8: Classical (0.94), Pop (0.83)

### Tensorflow (FFNN: i-Dense(128)-Dropout(.25)-Dense(784)-Dropout(.25)-Dense(784)-Dropout(.25)-Dense(10,softmax), Adagrad)

* Accuracy: 72%
* F1 scores over 0.8: Classical (0.94), Jazz (0.84) Pop (0.82), Hip-Hop (0.8)


## Conclusions

The Adagrad TensorFlow model was the most effective as a system to classify music genres since it had the largest amount of high F1 scores. If building a model to decide if a piece of music is classical or not, both TF models are more effective, but the Random Forest Classifier is more computationally efficient. The model needs to be scaled with a larger dataset to be created in the future. Data augmentation techniques can also be considered using libROSA
