from flask import Flask, request
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def hello():   
    filelist = os.listdir('static/Audio Files 2') 
    train_df = pd.DataFrame(filelist)
    train_df = train_df.rename(columns={0:'file'})
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    speaker = []
    for i in range(0, len(train_df)):
        speaker.append(train_df['file'][i][0:-5])
    train_df['speaker'] = speaker
    train_features = train_df.apply(extract_features, axis=1)
    features_test = []
    for i in range(0, len(train_features)):
        features_test.append(np.concatenate((train_features[i][0], train_features[i][1], 
                    train_features[i][2], train_features[i][3],
                    train_features[i][4]), axis=0))
    X_test = np.array(features_test)
    print(X_test)
    ss = StandardScaler()
    X_test = ss.fit_transform(X_test)
    return print(X_test)
    
def extract_features(files):
    file_name = os.path.join(os.path.abspath('static/Audio Files 2')+'/'+str(files.file))
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz
