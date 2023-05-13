import os
import librosa
import numpy as np
import warnings
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from matplotlib import pyplot 
from sklearn.preprocessing import LabelEncoder
import IPython.display as ipd
from keras.models import load_model
import random
import sounddevice as sd
import soundfile as sf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

#--------------------------------------------------------------------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
train_audio_path = 'C:\Users\ha_bo\OneDrive\Desktop\PFE\Dataset\train\audio'
labels=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

#--------------------------------------------------------------------------------------------------------------------------------------------

all_wave = []
all_label = []
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=8000)
        if(len(samples)== 8000) : 
            all_wave.append(samples)
            all_label.append(label)

le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

y=np_utils.to_categorical(y, num_classes=len(labels))

all_wave = np.array(all_wave).reshape(-1,8000,1)

#--------------------------------------------------------------------------------------------------------------------------------------------

x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)


K.clear_session()

inputs = Input(shape=(8000,1))

conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

conv = Flatten()(conv)

conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()

#--------------------------------------------------------------------------------------------------------------------------------------------

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('C:/Users/ha_bo/OneDrive/Desktop/PFE/models/best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#--------------------------------------------------------------------------------------------------------------------------------------------

history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

#--------------------------------------------------------------------------------------------------------------------------------------------

pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='test') 
pyplot.legend() 
pyplot.show()

#--------------------------------------------------------------------------------------------------------------------------------------------

model=load_model('C:/Users/ha_bo/OneDrive/Desktop/PFE/models/best_model.hdf5')

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

#--------------------------------------------------------------------------------------------------------------------------------------------

index=random.randint(0,len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:",predict(samples))

#--------------------------------------------------------------------------------------------------------------------------------------------

samplerate = 16000  
duration = 1
filename = 'C:/Users/ha_bo/OneDrive/Desktop/PFE/Commands/command.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)

filepath='C:/Users/ha_bo/OneDrive/Desktop/PFE/Commands'

samples, sample_rate = librosa.load(filepath + '/' + 'command.wav', sr = 16000)
samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=8000)
ipd.Audio(samples,rate=8000)  

predict(samples)

#--------------------------------------------------------------------------------------------------------------------------------------------

predictions = []
test = []
for i in range(0, 10):
    index=random.randint(0,len(x_val)-1)
    samples=x_val[index].ravel()
    print("Audio:",classes[np.argmax(y_val[index])])
    test.append(classes[np.argmax(y_val[index])])
    ipd.Audio(samples, rate=8000)
    print("Text:",predict(samples))
    predictions.append(predict(samples))
    
print(test)
print(predictions)

precision = precision_score(test, predictions, average='macro')
print("Precision:", precision)

recall = recall_score(test, predictions, average='macro')
print("Recall:", recall)

f1 = f1_score(test, predictions, average='macro')
print("F1 Score:", f1)

accuracy = accuracy_score(test, predictions)
print("Accuracy:", accuracy)



import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
confusion_mat = confusion_matrix(test, predictions)

# Get the unique class labels
class_labels = sorted(set(test))

# Plot confusion matrix with labels
sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------

predictions = []
for i in range(0, 10):
    samplerate = 16000  
    duration = 1
    filename = 'C:/Users/ha_bo/OneDrive/Desktop/PFE/Commands/command.wav'
    print('"say', labels[i], '"')
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    print("end")
    sd.wait()
    sf.write(filename, mydata, samplerate)

    filepath='C:/Users/ha_bo/OneDrive/Desktop/PFE/Commands'

    samples, sample_rate = librosa.load(filepath + '/' + 'command.wav', sr = 16000)
    samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=8000)
    ipd.Audio(samples,rate=8000)  
    predictions.append(predict(samples))
    
print(labels)
print(predictions)

precision = precision_score(labels, predictions, average='macro')
print("Precision:", precision)

recall = recall_score(labels, predictions, average='macro')
print("Recall:", recall)

f1 = f1_score(labels, predictions, average='macro')
print("F1 Score:", f1)

accuracy = accuracy_score(labels, predictions)
print("Accuracy:", accuracy)

confusion_mat = confusion_matrix(labels, predictions)
print("Confusion Matrix:")
print(confusion_mat)

