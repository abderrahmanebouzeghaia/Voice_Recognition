import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


train_audio_path = 'C:/Users/ha_bo/Downloads/tensorflow-speech-recognition-challenge/train/train/audio/'

#--------------------------------------------------------------------------------------------------------------------------------------------

samples, sample_rate = librosa.load(train_audio_path+'yes/0a7c2a8d_nohash_0.wav', sr = 16000)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + 'C:/Users/ha_bo/Downloads/tensorflow-speech-recognition-challenge/train/train/audio/yes/0a7c2a8d_nohash_0.wav')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

#--------------------------------------------------------------------------------------------------------------------------------------------

labels=os.listdir(train_audio_path)

no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------

labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


duration_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    
plt.hist(np.array(duration_of_recordings))
