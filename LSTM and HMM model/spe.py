import os
import random
import sys

import librosa
import numpy as np
from scipy.linalg import solve
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import normalize
from sklearn import linear_model

from confing import *
from cs_utils import CommonSpace, is_in_dict,getError
from decompose import decompose, decompose_with_dict
from getVectorAndPoint import (CalCommonSpacePoint, IndividualitySpace,
                               MixVectorAndIndividVector)
from utils import *
from confing import TIMESTEPS

class HMMPrepare():
    def __init__(self,mixSignal,sigAdict,sigBdict,commDist):
        self.__mixSignal = mixSignal
        self.__sigAdict = sigAdict
        self.__sigBdict = sigBdict
        self.__commDist = commDist
        self.__label = None
        self.__mixSTFT = None

    def __getState(self,epsilon = EPSILON):
        self.__mixSTFT = np.abs(get_spec(self.__mixSignal))
        column = self.__mixSTFT.shape[1]
        label = np.zeros([column])
        with open('result.txt','w') as ftp:
            for i in range(0,column):
                aErr = getError(self.__mixSTFT[:,i],self.__sigAdict)
                bErr = getError(self.__mixSTFT[:,i],self.__sigBdict)
                cErr = getError(self.__mixSTFT[:,i],self.__commDist)
                ftp.writelines(str([aErr,bErr,cErr]))
                #print([aErr,bErr,cErr])
                if cErr < epsilon:
                    label[i] = 0
                elif  aErr < epsilon :#A的独立部分
                    label[i] = 1
                elif bErr < epsilon:#B的独立部分
                    label[i] = 2
                else:#A的独立部分+B独立部分
                    label[i] = 3

            return label
    #获取状态后的处理---不妨假设HMM模型隐含状态1,2,3，观测状态1,2,3,4,5,6。根据采用维特比算法确定最优隐含状态序列
    def __getLable(self):
        if os.access('Label.txt',os.F_OK):
            self.__label = loadValue('Label.txt')[0]
        else:
            self.__label = self.__getState()
            dumpValue([self.__label],'Label.txt')
        
    def process(self):
        self.__getLable()
        return self.__label

class Fsovle():
    pass
    
def Initialization():
    trainFilelist = [r'dataspeech\train\track1.wav',r'dataspeech\train\track10.wav']
    testFilelist = [r'dataspeech\test\test.wav']
    signalA = readSignal(trainFilelist[0])
    signalB = readSignal(trainFilelist[1])
    testSignal = readSignal(testFilelist[0])
    
    signalASTFT = get_spec(signalA)
    signalBSTFT = get_spec(signalB)

    Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict = CalCommonSpacePoint(signalASTFT,signalBSTFT).getCommonSpace(k=SPEECH_RANK)
    dumpValue([Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal],'NMFinformation.txt')
    return Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal

def getdata():
    if os.access('NMFinformation.txt',os.F_OK): 
        Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal = loadValue('NMFinformation.txt')
    else:
        Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal =  Initialization()
    
    trainFilelist = [r'dataspeech\train\track1.wav',r'dataspeech\train\track10.wav']
    signalA = readSignal(trainFilelist[0])
    signalB = readSignal(trainFilelist[1])
    testSignal = signalA + signalB

    label = HMMPrepare(testSignal,sigAdict,sigBdict,Commondict).process()
    [sigAH,sigBH] = loadValue('signalH.txt')
    
    sepcindata = np.abs(get_spec(testSignal))
    specotdata = np.row_stack([sigAH,sigBH])
    size = label.shape[0]
    label = label.astype(np.int32)
    speclabel = np.zeros([4,size],dtype='float32')
    for i in range(0,size):
        r = int(label[i])
        speclabel[r,i] = 1

    m = size - size%TIMESTEPS
    sepcindata = sepcindata[:,0:m]
    specotdata = specotdata[:,0:m]
    speclabel = speclabel[:,0:m]
    
    n,m = sepcindata.shape
    sepcindata.resize([n,TIMESTEPS,int(m/TIMESTEPS)])

    n,m = specotdata.shape
    specotdata.resize([n,TIMESTEPS,int(m/TIMESTEPS)])

    n,m = speclabel.shape
    speclabel = speclabel.reshape([n,TIMESTEPS,int(m/TIMESTEPS)])

    return sepcindata,specotdata,speclabel,int(m/TIMESTEPS)

def gettestData():
    if os.access('NMFinformation.txt',os.F_OK): 
        Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal = loadValue('NMFinformation.txt')
    else:
        Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal =  Initialization()
    testFilelist = [r'dataspeech\test\test.wav']
    signal = readSignal(testFilelist[0])
    label = HMMPrepare(signal,sigAdict,sigBdict,Commondict).process()
    sepcindata = np.abs(get_spec(signal)) 
    specotdata = np.zeros([120,sepcindata.shape[1]])

    size = label.shape[0]
    label = label.astype(np.int32)
    speclabel = np.zeros([4,size],dtype='float32')
    for i in range(0,size):
        r = int(label[i])
        speclabel[r,i] = 1

    m = size - size%TIMESTEPS
    sepcindata = sepcindata[:,0:m]
    specotdata = specotdata[:,0:m]
    speclabel = speclabel[:,0:m]
    
    n,m = sepcindata.shape
    sepcindata=sepcindata.reshape([n,TIMESTEPS,int(m/TIMESTEPS)])

    n,m = specotdata.shape
    specotdata = specotdata.reshape([n,TIMESTEPS,int(m/TIMESTEPS)])

    n,m = speclabel.shape
    speclabel = speclabel.reshape([n,TIMESTEPS,int(m/TIMESTEPS)])

    return sepcindata,specotdata,speclabel,int(m/TIMESTEPS)

    

if __name__ == "__main__":
    #getdata()
    if os.access('NMFinformation.txt',os.F_OK): 
        Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal = loadValue('NMFinformation.txt')
    else:
        Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal =  Initialization()
    '''print(labelSigA.shape,np.sum(labelSigA))
    print(labelSigB.shape,np.sum(labelSigB))
    
    trainFilelist = [r'dataspeech\train\track1.wav',r'dataspeech\train\track10.wav']
    signalA = readSignal(trainFilelist[0])
    signalB = readSignal(trainFilelist[1])
    testSignal = signalA + signalB

    label = HMMPrepare(testSignal,sigAdict,sigBdict,Commondict).process()
    
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.figure(1)
    plt.hist(label)
    plt.figure(2)
    for i in range(label.shape[0]):
        if label[i] != 1.0 and label[i] != 2.0:
            label[i] = 3
    plt.hist(label)
    plt.show()
    '''

    '''testFilelist = [r'dataspeech\test\test.wav',r'dataspeech\test\track1.wav',r'dataspeech\test\track10.wav']
    signalA = readSignal(testFilelist[1])
    mixsignal = readSignal(testFilelist[0])
    signalASTFT = get_spec(signalA)
    mixsignalSTFT = get_spec(mixsignal)
    write_wav(spec2sig_from_sig(signalASTFT,mixsignal),'last1.wav')'''
    sepcindata,specotdata,speclabel,m = gettestData()
    lis = loadValue('spe.txt')
    #lis = specotdatau[:,:,0:100]
    #spe = lis[:,:,0]
    spe = None
    for i in range(0,len(lis)):
        for m in lis[i]:
            if spe is None:
                spe = m
            else:
                spe = np.column_stack([spe,m])

    '''trainFilelist = [r'dataspeech\train\track1.wav',r'dataspeech\train\track10.wav']
    signalA = readSignal(trainFilelist[0])
    signalB = readSignal(trainFilelist[1])
    testSignal = signalA + signalB
    N = FFT_LEN'''
    n,m,p = sepcindata.shape

    mixspe = sepcindata.reshape([n,m*p])
    mixspe = mixspe[:,0:spe.shape[1]]

    def spec2sig(spec, mix,mmix,sr=SAMPLE_RATE):
        N = FFT_LEN
        '''mask = np.abs(spec) / (np.abs(mmix)+0.000001)
        n,m = mask.shape
        for i in range(0,m):
            if np.sum(mask[:,i])>0.7:
                mask[:,i] = 1
            else:
                mask[:,i] = 0
        print(mask)
        signal_spec = np.array(mix) * np.array(mask)'''
        signal_spec = mix - spec
        sig = librosa.istft(signal_spec, hop_length=N // 2)
        return sig
    
    speA = np.dot(sigAdict,spe[0:60,:])
    speB = np.dot(sigBdict,spe[60:120,:])

    testFilelist = [r'dataspeech\test\test.wav']
    signal = readSignal(testFilelist[0])

    n,m = speA.shape
    sepcindata = get_spec(signal)
    mixspe = sepcindata[:,0:m]
    #mmix = np.abs(speA) + np.abs(speB)
    A = spec2sig(speA,mixspe,mixspe)
    write_wav(A ,'1.wav')
    #spec2sig(speB,mixspe,mmix)
    B = spec2sig(speB,mixspe,mixspe)
    write_wav(B,'2.wav')
    #write_wav(spec2sig(mmix,mixspe,mixspe),'3.wav')




    
    
    







