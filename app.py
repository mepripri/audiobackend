from flask import Flask, json, request
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from tensorflow import keras
app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def hello():
    return None

@app.route('/file', methods = ['GET', 'POST'])
def file():
    if(request.method=="POST"):
        user_file=request.files['file']
        path = os.path.join(os.path.abspath('static')+'/'+str(user_file.filename))
        user_file.save(path)
        train_df = pd.DataFrame(str(user_file.filename))
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
        ss = StandardScaler()
        X_test = ss.fit_transform(X_test)
        labels=open("labels.txt",'r')
        labels1=[]
        for i in labels:
            labels1.append(i[0:-1])
        labels1.pop()
        labels1.append("Yash")
        lb = LabelEncoder()
        y_train = to_categorical(lb.fit_transform(labels1))
        reconstructed_model = keras.models.load_model("model100.h5")
        predictions = reconstructed_model.predict_classes(X_test)
        predictions = lb.inverse_transform(predictions)
        print(predictions)

def extract_features(files):
    file_name = os.path.join(os.path.abspath('static')+'/'+str(files.file))
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    yield "doing"
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz
