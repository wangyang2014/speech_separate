import os
import numpy as np
import librosa
import sys
#sys.path.insert(0, '...')
SAMPLE_RATE = 22050
#from conf.confing import SAMPLE_RATE, FFT_LEN, CONSTANT
class SpeechSynthesis():
    def __init__(self,filePath = 'data\\TIMIT',personName = None):
        self.filePath = filePath
        self.personName = personName
        self.flielist = []

    def __getAllspeech(self):
        listpath = os.listdir(self.filePath)
        for path in  listpath:
            if path.lower() == "TEST".lower() or path.upper() == "TRAIN".upper() :
                self.__synthesis(self.filePath + '\\' + path)
            else:
                pass
    
    def __synthesis(self,path):
        for data in os.listdir(path):
            Name = path.split('\\')[-1]
            if os.path.isdir(path + '\\' + data):
                self.__synthesis(path + '\\' + data)
            else:
                output = 'F:\\' + path
                if not os.path.exists(output):
                    os.makedirs(output)
                filename = output + '\\' + Name + '.WAV'
                
                if os.path.isfile(filename):
                    print(filename)
                    continue

                signalALL = []
                for data in os.listdir(path):
                    if  '.WAV' in data:
                        signal, _ = librosa.load(path + '\\' + data, sr=SAMPLE_RATE)
                        signalALL = np.append(signalALL,signal)
                
                #signalALL = signalALL.astype(np.float)
                librosa.output.write_wav(filename, signalALL, SAMPLE_RATE, norm=True)
                self.flielist.append(filename)
                print(filename)

    def process(self):
        self.__getAllspeech()
        with open('speech synthesis flie path.txt','w') as ftp:
            for line in self.flielist:
                ftp.writelines(line) 

class Across():
    def __init__(self,signalA=None,signalB=None,fliePath_A=None,fliePath_B=None):
        self.mixSignal = None
        self.signalA = signalA
        self.signalB = signalB
        self.fliePath_A = fliePath_A
        self.fliePath_B = fliePath_B

    def __getMixSignal(self):
        if self.signalA is None:
            if self.fliePath_A is None:
                raise ValueError('the signalA is None')
            else:
                self.signalA, _ = librosa.load(self.fliePath_A, sr=SAMPLE_RATE)
        
        if self.signalB is None:
            if self.fliePath_B is None:
                raise ValueError('the signalB is None')
            else:
                self.signalB, _ = librosa.load(self.fliePath_B, sr=SAMPLE_RATE)

        n = self.signalA.shape
        n2 = self.signalB.shape
        N = int(min(n[0],n2[0]))
        #M = min(m,m2)

        data1 = self.signalA[0:N]
        data2 = self.signalB[0:N]
        self.mixSignal = data1 + data2

    def process(self):
        self.__getMixSignal()
        return self.mixSignal

        
if __name__ == '__main__':
    SpeechSynthesis().process()











'''
import wave as we
import numpy as np
import matplotlib.pyplot as plt

def wavread(path):
    wavfile =  we.open(path,"rb")
    params = wavfile.getparams()
    framesra,frameswav= params[2],params[3]
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav,dtype = np.int32)
    datause.shape = -1,2
    datause = datause.T
    time = np.arange(0, frameswav) * (1.0/framesra)
    return datause,time

def main():
    path = input("The Path is:")
    wavdata,wavtime = wavread(path)
    plt.title("Night.wav's Frames")
    plt.subplot(211)
    plt.plot(wavtime, wavdata[0],color = 'green')
    plt.subplot(212)
    plt.plot(wavtime, wavdata[1])
    plt.show()
    
main()'''