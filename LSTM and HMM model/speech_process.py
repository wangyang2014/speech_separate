import numpy as np
import sys
import os
import pickle
import random 
from utils import *
from cs_utils import *
from confing import *
from getVectorAndPoint import *

def  Vads(signal,size,threshold):
    Template = []
    for i in range(0,size):
        Template.append(signal[i])    
    for i in range(size,signal.shape[0]- size):
        data = signal[i-size:i+size]
        value = np.sum(np.abs(data))
        if value >= threshold:
            Template.append(signal[i])
    
    for i in range(signal.shape[0]- size,signal.shape[0]):
        Template.append(signal[i])

    return np.asarray(Template)

	
def randomMixSignal(signalA,signalB,frameLength):
	mixSig = None
	Asig = None
	Bsig = None
	n = int(signalA.shape[0]/frameLength)-1
	m = int(signalB.shape[0]/frameLength)-1
	i=j=0
	label = []
	while(i<n or j<m):
		dr = random.uniform(0,1)<1/2
		if dr<1/2:
			if i<n:
				inA = signalA[i*frameLength:(i+1)*frameLength]
				inB = np.zeros([frameLength])* random.uniform(0,0.00001)
				inM = inA
				i = i + 1
				label.append(1)
			else:
				continue
		elif dr>=1/2:
			if j<m:
				inA = np.zeros([frameLength]) * random.uniform(0,0.00001)
				inB = signalB[j*frameLength:(j+1)*frameLength]
				inM = inB
				j = j + 1
				label.append(2)
			else:
				continue
		else:
			label.append(3)
			inA = np.zeros([frameLength])* random.uniform(0,0.00001)
			inM = inB = inA
		
		#print(inA.shape,inB.shape,inM.shape)

		if 	Asig is None:
			Asig = inA
		else:
			try:
				Asig = np.row_stack([Asig,inA])
			except:
				print(n,m,i,j)
				data = len(inA)
				print(Asig.shape) 
			#np.column_stack()
		if Bsig is None:
			Bsig = inB
		else:
			Bsig = np.row_stack([Bsig,inB])
		
		if mixSig is None:
			mixSig = inB
		else:
			mixSig = np.row_stack([mixSig,inM])
			
	return mixSig,Asig,Bsig,label
	
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

def getTextsignal():
	testFilelist = [r'dataspeech\test\track1.wav',r'dataspeech\test\track10.wav',r'dataspeech\test\test.wav']
	signalA = readSignal(testFilelist[0])
	signalB = readSignal(testFilelist[1])
	
	signalA =  Vads(signalA,64,0.25)
	signalB =  Vads(signalB,64,0.25)
	write_wav(signalA,'signalAvad.wav')
	n,m = signalA.shape[0],signalB.shape[0]

	mixSig = np.zeros([n+m],dtype=np.float32)
	Asig = np.zeros([n+m],dtype=np.float32)
	Bsig = np.zeros([n+m],dtype=np.float32)

	mixSig[0:n] = signalA
	Asig[0:n] = signalA

	mixSig[n:m+n] = signalB
	Bsig[n:m+n] = signalB
	#mixSig,Asig,Bsig,label = randomMixSignal(signalA,signalB,2560*5)
	#mixSig = mixSig.reshape([-1,])
	#Asig = Asig.reshape([-1,])
	#Bsig = Bsig.reshape([-1,])
	write_wav(mixSig,'mixSig.wav')
	write_wav(Asig,'Asig.wav')
	write_wav(Bsig,'Bsig.wav')
	return mixSig,Asig,Bsig
	
	
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
                #cErr = getError(self.__mixSTFT[:,i],self.__commDist)
                ftp.writelines(str([aErr,bErr]))
                #print([aErr,bErr,cErr])
                if  aErr < epsilon and bErr < epsilon:#A的独立部分
                    label[i] = 3
                elif bErr < aErr:#B的独立部分
                    label[i] = 2
                else:#不可分
                    label[i] = 1

            return label
    #获取状态后的处理---不妨假设HMM模型隐含状态1,2,3，观测状态1,2,3,4,5,6。根据采用维特比算法确定最优隐含状态序列
    def __getLable(self):
        if os.access('Labelr.txt',os.F_OK):
            self.__label = loadValue('Label.txt')[0]
        else:
            self.__label = self.__getState()
            dumpValue([self.__label],'Label.txt')
        
    def process(self):
        self.__getLable()
        return self.__label	
		
def dealLabel(Lable):
	return Lable
	
if __name__ == "__main__":
	#Asig=readSignal('Asig.wav')
	'''mixSig,Asig,Bsig = getTextsignal()

	if os.access('NMFinformation.txt',os.F_OK):
		Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal = loadValue('NMFinformation.txt')
	else:
	    Commondict,_,labelSigA,labelSigB,sigAdict,sigBdict,testSignal =  Initialization()
	
	testSignal = mixSig
	Lable = HMMPrepare(testSignal,sigAdict,sigBdict,None).process()
	Lable = dealLabel(Lable)
	mixspe = get_spec(testSignal)
	sigA = np.zeros(mixspe.shape)
	sigB = np.zeros(mixspe.shape)
	
	flagA = Lable==1
	sigA[:,flagA] =  1
	sigAspe = sigA * mixspe
	
	flagB = Lable==2
	sigB[:,flagB] =  1
	sigBspe = sigB * mixspe
	
	N = FFT_LEN
	Asig = librosa.istft(sigAspe, hop_length=N // 2)
	Bsig = librosa.istft(sigBspe, hop_length=N // 2)
	
	write_wav(Asig,'Asignew.wav')
	write_wav(Bsig,'Bsignew.wav')
	'''
	signal = readSignal('signalAvad.wav')
	specA = get_spec(signal)
	n,m = specA.shape
	Label = loadValue('Label.txt')[0]

	L = Label.shape[0]

	print((np.sum(Label[0:m]) - m)/m , (np.sum(Label[m:L]) - (L-m))/(L-m))
	Lm = np.zeros([L])
	Lm[0:m] = 1
	Lm[m:L] = 2
	
	print(1 - np.sum(np.abs(Lm - Label))/L)

