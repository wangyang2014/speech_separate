import numpy as np 
from utils import *
def getTrainData(filepath,rata = 0.95):
    signal = readSignal(filepath)
    size = int(signal.shape[0] * rata)
    return signal[0:size]

def getTestData(filepath,rata = 0.95):
    signal = readSignal(filepath)
    size = int(signal.shape[0] * rata)
    return signal[size+1:]

if __name__ == "__main__":
    #signal = readSignal(r'dataspeech\train\track1.wav')
    signal = getTrainData(r'dataspeech\track1.wav')
    wirteSignal(signal,r'dataspeech\train\track1.wav')
    signalA = getTestData(r'dataspeech\track1.wav')
    wirteSignal(signalA,r'dataspeech\test\track1.wav')

    signal = getTrainData(r'dataspeech\track10.wav')
    wirteSignal(signal,r'dataspeech\train\track10.wav')
    signalB = getTestData(r'dataspeech\track10.wav')
    wirteSignal(signalB,r'dataspeech\test\track10.wav')
    wirteSignal(signalA + signalB,r'dataspeech\test\test.wav')
