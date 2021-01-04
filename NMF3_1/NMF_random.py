import os
import numpy as np
import librosa
import sys
import math

from confing import SAMPLE_RATE, FFT_LEN, CONSTANT,SPEECH_RANK
from decompose import decompose, decompose_with_dict
from utilsm import *

# 10log(10) A/B  = SNR --> B = A/(10^(SNR/10))
def Generate(sigA,sigB,SNR=0):
    #alignment
    size = np.min([sigA.shape[0],sigB.shape[0]])
    sigA = sigA[:size]
    sigB = sigB[:size]
    # 获取比例系数
    energyA = np.sum(sigA**2)
    energyB = np.sum(sigB**2)
    energyb = energyA/(10**(SNR/10))
    corr = math.sqrt(energyb/energyB)
    # mix 
    sigB = sigB * corr
    mix = sigA + sigB
    #不能越界
    value = np.max(np.abs(mix))
    if value > 1:
        weight = 1/value
        mix = mix*weight
        sigA = sigA*weight
        sigB = sigB*weight
    
    return sigA,sigB,mix

class NMF():
    def __init__(self,signal):
        self.__signalASTFT = get_spec(signal)
    def getVectorandWeight(self):
        vector,weight = decompose(np.abs(self.__signalASTFT), k=SPEECH_RANK)
        return vector,weight
    def getWeight(self,vector):
        weight = decompose_with_dict(np.abs(self.__signalASTFT),vector)
        return weight

def Initialization(filelist):
    signalA = readSignal(filelist[0])
    signalB = readSignal(filelist[1])
    sigA,sigB,mix = Generate(signalA,signalB)
    V_A,_ = NMF(sigA).getVectorandWeight()
    V_B,_ = NMF(sigB).getVectorandWeight()

    commonVec = np.column_stack([V_B,V_A])
    weight = NMF(mix).getWeight(commonVec)
    return V_A,V_B,weight,mix,sigA,sigB

if __name__ == "__main__":
    filelist = [r"C:\Users\Lala\Desktop\NMF\data\FAEM0\SA1_.wav",r"C:\Users\Lala\Desktop\NMF\data\FCAJ0\SX129_.wav"]
    V_A,V_B,weight,mix,sigA_s,sigB_s = Initialization(filelist)

    sigA = np.dot(V_A,weight[SPEECH_RANK:])
    sigB = np.dot(V_B,weight[:SPEECH_RANK])
    
    signal_spec = get_spec(mix)

    #Wiener-Type Filtering
    sigA_spec = signal_spec*(sigA/(sigA+sigB + 1e-8))
    sinB_spec = signal_spec*(sigB/(sigA+sigB + 1e-8))


    sigA = librosa.istft(sigA_spec, hop_length=FFT_LEN // 2)
    sigB = librosa.istft(sinB_spec, hop_length=FFT_LEN // 2)


    energyA = np.sum(sigA**2)
    energyB = np.sum(sigB**2)

    print(energyA,energyB)
    print(energyB/(energyA+energyB))

    import matplotlib.pyplot as plt
    length = np.min([sigA.shape[0],sigA_s.shape[0],sigB_s.shape[0],mix.shape[0]])
    sigA = sigA[:length]
    sigA_s = sigA_s[:length]
    sigB = sigB[:length]
    sigB_s = sigB_s[:length]
    mix = mix[:length]
    time = np.arange(0,sigB.shape[0]) * (1.0 / SAMPLE_RATE)

    plt.figure()

    plt.subplot(5,2,1)
    plt.plot(time,mix,c="b")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("mix wave")

    plt.subplot(5,2,3)
    plt.plot(time,sigA_s,c="b")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("speaker 1 Original wave")

    plt.subplot(5,2,5)
    plt.plot(time,sigA,c="b")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("speaker 1 NMF out wave")

    plt.subplot(5,2,7)
    plt.plot(time,sigB_s,c="b")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("speaker 2 Original wave")

    plt.subplot(5,2,9)
    plt.plot(time,sigB,c="b")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("speaker 2 NMF out wave")

    plt.subplot(5,2,2)
    spectrum,freqs,ts,fig = plt.specgram(mix,NFFT = FFT_LEN,Fs =SAMPLE_RATE,window=np.hanning(M = FFT_LEN),noverlap=FFT_LEN//2,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title("mix Spectrogram")
   
    plt.subplot(5,2,4)
    sigAm = sigA * 0
    spectrum,freqs,ts,fig = plt.specgram(sigA_s,NFFT = FFT_LEN,Fs =SAMPLE_RATE,window=np.hanning(M = FFT_LEN),noverlap=FFT_LEN//2,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title("speaker 1 Original wave Spectrogram")

    plt.subplot(5,2,6)
    spectrum,freqs,ts,fig = plt.specgram(sigA,NFFT = FFT_LEN,Fs =SAMPLE_RATE,window=np.hanning(M = FFT_LEN),noverlap=FFT_LEN//2,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title("speaker 1 NMF out wave Spectrogram")
    
    plt.subplot(5,2,8)
    spectrum,freqs,ts,fig = plt.specgram(sigB_s,NFFT = FFT_LEN,Fs =SAMPLE_RATE,window=np.hanning(M = FFT_LEN),noverlap=FFT_LEN//2,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title("speaker 2 Original wave Spectrogram")
    
    plt.subplot(5,2,10)
    spectrum,freqs,ts,fig = plt.specgram(sigB_s,NFFT = FFT_LEN,Fs =SAMPLE_RATE,window=np.hanning(M = FFT_LEN),noverlap=FFT_LEN//2,mode='default',scale_by_freq=True,sides='default',scale='dB',xextent=None)#绘制频谱
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title("speaker 2 NMF out wave Spectrogram")

    plt.show()