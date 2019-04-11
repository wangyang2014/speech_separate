import os
import numpy as np
import librosa
import sys
from confing import SAMPLE_RATE, FFT_LEN, CONSTANT,SPEECH_RANK
from spesyn import SpeechSynthesis ,Across 
from decompose import decompose, decompose_with_dict
from cs_utils import CommonSpace,is_in_dict
from utilsm import get_spec, norm_signal, mix_noise, spec2sig_from_sig, write_wav, cal_pesq, get_stats

class MixVectorAndIndividVector():
    def __init__(self,signalA=None,signalB=None):
        self.__signalASTFT = get_spec(signalA)
        self.__signalBSTFT = get_spec(signalB)
        self.__mixVector = None
        self.__IndividVector = None

    def __getDict (self):
        sigADict,W_a = decompose(np.abs(self.__signalASTFT), k=SPEECH_RANK)
        sigBdict,W_b = decompose(np.abs(self.__signalBSTFT), k=SPEECH_RANK)
        return sigADict,sigBdict,W_a,W_b
    
    def __getVector(self,dictoray,sigSpec):
        _,m = sigSpec.shape
        for i in range(0,m):
            vector = sigSpec[:,i]
            #print(vector,type(vector))
            if is_in_dict(vector,dictoray):
                if self.__mixVector is None:
                    self.__mixVector = vector
                else:
                    self.__mixVector = np.column_stack([self.__mixVector,vector])
            else:
                if self.__IndividVector is None:
                    self.__IndividVector = vector
                else:
                    self.__IndividVector = np.column_stack([self.__IndividVector,vector])

    def process(self):
        sigADict,sigBdict,W_a,W_b = self.__getDict()
        self.__getVector(sigBdict,sigADict)
        self.__getVector(sigADict,sigBdict)
        return self.__IndividVector,W_a,W_b,sigADict,sigBdict


    def setSignal(self,signalA=None,signalB=None):
        self.__signalASTFT = get_spec(signalA)
        self.__signalBSTFT = get_spec(signalB)
        self.__mixVector,self.__IndividVector = None,None


class CalCommonSpacePoint():
    def __init__(self,signalA=None,signalB=None):
        self.__signalASTFT = get_spec(signalA)
        self.__signalBSTFT = get_spec(signalB)
        self.__commonSpacePoint = None
        self.__commonSpaceVector = None
        self.__mixVector = None
        self.__IndividSpace = None
    
    def __getDict (self):
        sigADict,_ = decompose(np.abs(self.__signalASTFT), k=SPEECH_RANK)
        sigBdict,_ = decompose(np.abs(self.__signalBSTFT), k=SPEECH_RANK)
        return sigADict,sigBdict

    def __getPoint(self,dictoray,sigSpec):
        _,m = sigSpec.shape
        label = np.zeros([m])
        for i in range(0,m):
            vector = sigSpec[:,i]
            if is_in_dict(vector,dictoray):
                if self.__commonSpacePoint is None:
                    self.__commonSpacePoint = vector
                else:
                    self.__commonSpacePoint = np.column_stack([self.__commonSpacePoint,vector])
                label[i] = 1
        return label

    def __getCommonSpacePoint(self):
        sigAdict,sigBdict = self.__getDict()
        labelSigA = self.__getPoint(sigBdict,np.abs(self.__signalASTFT))
        labelSigB = self.__getPoint(sigAdict,np.abs(self.__signalBSTFT))
        return sigAdict,sigBdict

    def getCommonSpacePoint(self):
        self.__getCommonSpacePoint()
        return self.__commonSpacePoint

    def getgetCommonSpace(self,k, max_iter=6000, alpha=0.5):
        sigAdict,sigBdict = self.__getCommonSpacePoint()
        W,H = decompose(self.__commonSpacePoint,k,max_iter,alpha)
        return W,H,sigAdict,sigBdict

    def setSignal(self,signalA=None,signalB=None):
        self.__signalASTFT = get_spec(signalA)
        self.__signalBSTFT = get_spec(signalB)
        self.__commonSpacePoint = None


class IndividualitySpace():
    def __init__(self,signal):
        self.__signal = signal
        self.__individDict = None

    def __getIndividDict(self):
        value = get_spec(self.__signal)
        self.__individDict,_ = decompose(np.abs(value), k=SPEECH_RANK)

    def setSignal(self,signal):
        self.__signal = signal 

    def getDict(self):
        self.__getIndividDict()
        return self.__individDict