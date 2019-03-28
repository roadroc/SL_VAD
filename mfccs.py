# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:43:37 2018

@author: roadr
"""

import numpy, scipy, sklearn
import librosa, librosa.display, matplotlib.pyplot as plt

x, fs = librosa.load('C:/Users/roadr/Documents/Python Scripts\Voice\D4_750.wav',sr=16000)
mfccs = librosa.feature.mfcc(x, sr=fs)
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
#mfccs.sort(axis=0)
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveplot(x, sr=fs)
plt.subplot(3, 1, 2)
librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.subplot(3, 1, 3)
mfccs.sort(axis=0) 
librosa.display.specshow(mfccs, sr=fs, x_axis='time')
plt.show()
#scipy.io.savemat('mfccs.mat',{'data_s':mfccs})