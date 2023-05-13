import os
import sounddevice as sd
import soundfile as sf

words = ["نعم", "لا", "أعلى", "أسفل", "يسار", "يمين", "على", "إيقاف", "توقف", "انطلق" , "igh", "alla", "udam", "azrar", "aseggas", "taslit", "alas", "iway", "asedkan", "aseɣɣed"]

data_dir = 'C:/Users/ha_bo/OneDrive/Desktop/PFE/Dataset_ML/'

duration = 1
samplerate = 16000 

for word in words:
    num_files_per_word = int(input(f'Enter the number of audio files to create for the word : "{word}" '))
    word_dir = os.path.join(data_dir, word)
    os.makedirs(word_dir, exist_ok=True)
    for i in range(num_files_per_word):
        filename = f'{word}_{i+1}.wav'
        file_path = os.path.join(word_dir, filename)
        input(f"Press Enter to record the {i+1} audio of the word -{word}-")
        mydata = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
        sd.wait()
        sf.write(file_path, mydata, samplerate)