from pypesq import pesq
import soundfile as sf
import librosa
SAMPLE_RATE = 16000
from utils import get_spec, norm_signal, mix_noise, spec2sig_from_sig, write_wav
def readSignal(fliepath,sr = SAMPLE_RATE):
    signal, _ = librosa.load(fliepath,sr=sr,mono=True)
    return signal
def wirteSignal(signal,filename):
    write_wav(signal,filename,sr=SAMPLE_RATE)
    
ref = readSignal('track1.mp3',16000)
deg = readSignal('track1New.mp3',16000)
n = min(ref.shape[0],deg.shape[0])

wirteSignal(ref[0:n],'track1.wav')
wirteSignal(deg[0:n],'track1New.wav')

ref = readSignal('track10.mp3',16000)
deg = readSignal('track10New.mp3',16000)
n = min(ref.shape[0],deg.shape[0])

wirteSignal(ref[0:n],'track10.wav')
wirteSignal(deg[0:n],'track10New.wav')