import os
import time
import wave
import pyaudio
import librosa
import numpy as np
import simpleaudio as sa

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioTrainTest as aT
import matplotlib.pyplot as plt
from sklearn import svm

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FILE_NUMBER = 1
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

if not os.path.exists('sounds'):
    os.makedirs('sounds')
if not os.path.exists('Male'):
    os.makedirs('Male')
if not os.path.exists('Female'):
    os.makedirs('Female')


def main():
    # def train():
    input('Speak for 5 secs after pressing \'Enter\': ')
    print('\nRecording')

    time.sleep(.5)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print('\nRecording Saved.')
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open('sounds/' + 'output%d.wav' % FILE_NUMBER, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # >>>>>>FEATURE EXTRACTION
    [fs, x] = audioBasicIO.readAudioFile('sounds/output%d.wav' % FILE_NUMBER)
    f, f_names = ShortTermFeatures.feature_extraction(x, fs, 0.050 * fs, 0.025 * fs)
    print(f_names)
    print(f)

    # def trainClassifier():
    # >>>>>TRAINING SVM
    aT.featureAndTrain(["Male/", "Female/", ], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm",
                       "svm2Classes")

    aT.fileClassification('sounds/output1.wav', "svm2Classes", svm)

    # def playAudio ():
    # Play audio
    input('To play audio press \'Enter\': '
          )
    filename = 'sounds/output1.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing
    print("Audio has finished playing")

    # def manipulate():
    [fs, x] = audioBasicIO.readAudioFile('sounds/output%d.wav' % FILE_NUMBER)
    f, f_names = ShortTermFeatures.feature_extraction(x, fs, 0.050 * fs, 0.025 * fs)
    input('To manipulate input press \'Enter\': '
          )
    # Create an array of random numbers to use as the adversarial input
    r = np.random.rand(68, 198)
    print("Adversarial input\n", r)

    # Create an empty array to allow the user to edit any feature they want.
    s = (68, 198)
    e = np.zeros(s)
    print("Empty data\n", e)

    # Print the feature values for the original audio clip
    print("Audio clip\n", f)

    # Multiply the original audio with manipulated data to see if it can misclassify
    m = f * r

    print("Manipulated data\n", m)

    # def plotGraphs ():
    # Plotting original input
    plt.subplot(2, 2, 1);
    plt.plot(f[0, :]);
    plt.xlabel('Original');
    plt.ylabel(f_names[0])

    # Plotting adversarial input
    plt.subplot(2, 2, 2);
    plt.plot(r[0, :]);
    plt.xlabel('Adversarial input');

    # Plotting manipulated data
    plt.subplot(2, 2, 3);
    plt.plot(m[0, :]);
    plt.xlabel('manipulated data');
    plt.show()

    # Convert manipulated array back into wav
    librosa.feature.inverse.mfcc_to_audio(m, n_mels=128, dct_type=2,norm='ortho', ref=1.0, lifter=0, **kwargs);


# select = input('Press 1 to train classifier model \'1\' \n'
#    'Press 2 to classify input \'2\' \n'
#     'Press 3 to manipulate input\'3\': ')

# if select == 1:
#  train()

# select == 2:
# classify
# if select == 3:
#   manipulate()


if __name__ == '__main__':
    main()
