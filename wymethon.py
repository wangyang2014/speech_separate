import os
import random
import sys

import librosa
import numpy as np
from scipy.linalg import solve
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import normalize

from confing import CONSTANT, FFT_LEN, SAMPLE_RATE, SPEECH_RANK
from cs_utils import CommonSpace, is_in_dict
from decompose import decompose, decompose_with_dict
from getVectorAndPoint import (CalCommonSpacePoint, IndividualitySpace,
                               MixVectorAndIndividVector)
from spesyn import Across, SpeechSynthesis
from utilsm import *


class ClustersInformation():
    def __init__(self,data):
        self.centre = None
        self.label = None
        self.data = normalize(data,norm = 'l2',axis=0)
        self.n_clusters = 100

    def __clusters(self,data,n_clusters = 100):
        data = np.transpose(data)
        #data  = normalize(data,norm = 'l2',axis=1)#z正则化。
        linkage,connectivity,n_clusters,affinity = 'average',None,n_clusters,'cosine'
        model = AgglomerativeClustering(linkage=linkage,
                                        connectivity=connectivity,
                                        n_clusters=n_clusters,
                                        affinity = affinity)
        model.fit(data)

        lable = model.labels_
        centre = {}
        for i in range(0,n_clusters):
            flag = lable == i
            meanvalue = np.mean(data[flag,:],axis=0)
            centre[str(i)] = meanvalue 

        return lable,centre
        
    def process(self):
        self.label,self.centre = self.__clusters(self.data)

    def getBestpoint(self,point,clu):
        index = -1
        maxvalue = 0
        flag = self.label == clu
        setdata = self.data[:,flag]

        _,m = setdata.shape
        for i in range(0,m):
            value = cosine_similarity(setdata[:,i].reshape(1, -1),point.reshape(1, -1))[0][0]
            if value > maxvalue:
                maxvalue = value
                index = i
        
        return setdata[:,index]
        
    
class Fsovle():
    def __init__(self,sigAVec,sigBVec,mixVec,W_mix,W_siga,W_sigb):
        self.__sigAVec = normalize(sigAVec,norm = 'l2',axis=0)
        self.__sigBVec = normalize(sigBVec,norm = 'l2',axis=0)
        #self.__sigBVec = sigBVec
        self.__mixVec = mixVec
        self.__w_Mix = W_mix
        self.__w_Siga = W_siga
        self.__w_Sigb = W_siga
        self.__A = np.dot(np.linalg.inv(np.dot(np.transpose(self.__sigAVec),self.__sigAVec)),np.transpose(self.__sigAVec))
        self.__B = np.dot(np.linalg.inv(np.dot(np.transpose(self.__sigBVec),self.__sigBVec)),np.transpose(self.__sigBVec))

    

    def train(self):
        #mixClusInf = self.__clusters()
        EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo = self.__statistics()
        return EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo

      
    def __statistics(self):
        _,m = self.__w_Mix.shape
        mixInfo = ClustersInformation(self.__w_Mix)
        mixInfo.process()
        sigAInfo = ClustersInformation(self.__w_Siga)
        sigAInfo.process()
        sigBInfo = ClustersInformation(self.__w_Sigb)
        sigBInfo.process()
        
        size = sigAInfo.n_clusters
        #print(size)
        frequencyWithSingnalA = np.zeros([size,size])
        frequencyWithSingnalB = np.zeros([size,size])
        
        for i in range(0,m):
            frequencyWithSingnalA[mixInfo.label[i],sigAInfo.label[i]] += 1
            frequencyWithSingnalB[mixInfo.label[i],sigBInfo.label[i]] += 1

        EPMix2sigA = np.transpose(np.transpose(frequencyWithSingnalA)/np.sum(frequencyWithSingnalA,axis=1))#emissionProbability
        EPMix2sigB = np.transpose(np.transpose(frequencyWithSingnalB)/np.sum(frequencyWithSingnalB,axis=1))#emissionProbability

        return EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo
    
    def __getClusters(self,sigInfo,wight):
        m = sigInfo.n_clusters
        ESP = 10 ** -10
        distance = []
        for i in range(0,m):
            data = sigInfo.centre[str(i)]
            
            value = cosine_similarity(data.reshape(1, -1)+ESP,wight.reshape(1, -1)+ESP)[0][0]
            distance.append(value)
        
        return distance

    def __getBestWight(self,EPMix2sig,index,wight,sigInfo):
        ep = EPMix2sig[index]  #Emission probability
        distance =  np.asarray(self.__getClusters(sigInfo,wight))
        info = ep * distance

        indexClu = np.where(info ==  np.max(info))[0][0]
        #bestWight = sigInfo.getBestpoint(wight,indexClu) 
        bestWight = sigInfo.centre[str(indexClu)]

        return bestWight

    def eM_method(self,EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo,mixWight,max_iter= 10):
        mixDist = self.__getClusters(mixInfo,mixWight)
        index = mixDist.index(max(mixDist))
        indexM = np.where(EPMix2sigA[index,:] ==  np.max(EPMix2sigA[index,:]))[0][0]
        
        bestAWight = sigAInfo.centre[str(indexM)] #np.random.uniform(0,100,1) * sigAInfo.centre[indexM]
        M = np.dot(self.__mixVec,mixWight) 
        Bi = 10
        Ai = 10
        for i in range(0,max_iter): 
            #ESP = 10**10
            #E step
            Yvalue = np.dot(self.__sigAVec,bestAWight)
            #value1 = min(np.min(M/Yvalue),1) #np.linalg.norm(bestAWight,2)#min(np.min(M/Yvalue),1)
            b = M - Yvalue #* value1
            #b = M - value *  Yvalue
            bWight = np.dot(self.__B,b)#decompose_with_dict(b.reshape(-1, 1) ,self.__sigBVec)#np.dot(self.__B,b) # 最小二乘法
            #bWight = bWight.reshape(1,- 1)[0,:]
            if random.uniform(0,1) < 1:
                bestBWight = self.__getBestWight(EPMix2sigB,index,bWight,sigBInfo)
                #Bi -= 1
                bestBWight =( np.linalg.norm(bWight,2) * bestBWight + bWight )/2
            else:
                bestBWight = bWight
            #保证权重为正数\
            Yvalue = np.dot(self.__sigBVec,bestBWight)
            value2 = min(np.min(M/Yvalue),1)#np.linalg.norm(bWight,2)#min(np.min(M/Yvalue),np.linalg.norm(bWight,2))
            #bestBWight = value * bestBWight
            
            #M step
            b = M - Yvalue #* value2#np.dot(self.__sigBVec,bestBWight)
            aWight = np.dot(self.__A,b)#decompose_with_dict(b.reshape(-1, 1) ,self.__sigAVec)
            #aWight = aWight.reshape(1,- 1)[0,:]
            if  random.uniform(0,1) < 1:
                bestAWight = self.__getBestWight(EPMix2sigA,index,aWight,sigAInfo)
                #Ai -= 1
                bestAWight = (np.linalg.norm(aWight,2) * bestAWight + aWight)/2
            else:
                bestAWight = aWight
           
            #value = np.min(M/np.dot(self.__sigAVec,bestAWight))
             #min(np.linalg.norm(aWight,2),value) * bestAWight
            if i % 100 == 0:
                print(i)

        return bestAWight,bestBWight

 
def Initialization():
    flielist = [r'dataspeech\track1.wav',r'dataspeech\track10.wav',r'dataspeech\across.wav']
    signalA = readSignal(flielist[0])
    signalB = readSignal(flielist[1])
    mixSignal = readSignal(flielist[2])

    IndividVector,W_siga,W_sigb,sigADict,sigBdict = MixVectorAndIndividVector(signalA,signalB).process()
    
    specSignal = get_spec(mixSignal) 
    W_mix = decompose_with_dict(np.abs(specSignal),IndividVector)

    return IndividVector,W_mix,sigADict,W_siga,sigBdict,W_sigb
'''
import threading
def getBest(i):
    mixWight = W_mix[:,i]
    bestAWight,bestBWight =  Ftest.eM_method(EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo,mixWight)
    signalA.append([bestAWight,i])
    signalB.append([bestBWight,i])

count = 20500
class myThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
    
    def run(self):
        print ("开始线程：" + self.name)
        while(True):
            threadLock.acquire()
            global  count
            count = count + 1
            threadLock.release()
            if count < 21000:
                getBest(count)
            else:
                break
'''
        
if __name__ == "__main__":
    if os.access('vectorAndWight.txt',os.F_OK):
        IndividVector,W_mix,sigADict,W_siga,sigBdict,W_sigb= loadValue('vectorAndWight.txt')    
    else:
        IndividVector,W_mix,sigADict,W_siga,sigBdict,W_sigb = Initialization()
        dumpValue([IndividVector,W_mix,sigADict,W_siga,sigBdict,W_sigb],'vectorAndWight.txt')
    
    Ftest = Fsovle(sigADict,sigBdict,IndividVector,W_mix[:,0:20000],W_siga[:,0:20000],W_sigb[:,0:20000])
    
    if os.access('trainData.txt',os.F_OK):
        EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo = loadValue('trainData.txt')
        print('...')
    else:
        EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo = Ftest.train()
        dumpValue([EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo],'trainData.txt')
    
    signalA = []
    signalB = []
    #threadLock = threading.Lock()
    if os.access('signal.txt',os.F_OK):
        pass
    else:
        for i in range(20000,20500):
            mixWight = W_mix[:,i]
            bestAWight,bestBWight =  Ftest.eM_method(EPMix2sigA,EPMix2sigB,mixInfo,sigAInfo,sigBInfo,mixWight)
            signalA.append(bestAWight)
            signalB.append(bestBWight)
            print(i)
        dumpValue([signalA,signalB],'NMFMESsignal.txt')

    '''
    threadlist = []
    
    for i in range(0,10):
        thread = myThread(i, "Thread-0" + str(i))
        threadlist.append(thread)
    for thread in threadlist:
        thread.start()
    for thread in threadlist:
        thread.join()

    dumpValue([signalA,signalB],'signalaThread.txt')

    print(len(signalA),len(signalB))'''
    mixSPE = np.dot(IndividVector,W_mix[:,20000:20500])

    signalA,signalB = loadValue('NMFMESsignal.txt')
    sigA = np.zeros([60,500])
    sigB = np.zeros([60,500])
    
    for i in range(0,500):
        sigA[:,i] = signalA[i]
        sigB[:,i] = signalB[i]
    
    partA = np.dot(sigADict,sigA)
    #partA = (partA - np.min(partA))/(np.max(partA) - np.min(partA))
    partB = np.dot(sigBdict,sigB)
    V = partA + partB
    mixSignal = readSignal(r'dataspeech\across.wav')[20000*FFT_LEN:FFT_LEN*20250]
    write_wav(spec2sig_from_sig(partA,mixSignal),'track1Newrand.wav')
    write_wav(spec2sig_from_sig(partB/np.max(partB),mixSignal),'track10Newfastrand.wav')
