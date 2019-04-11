from pydub.audio_segment import AudioSegment
inputflie = [r'C:\Users\Administrator\Desktop\wyspeechcode\data\track1.mp3',r'C:\Users\Administrator\Desktop\wyspeechcode\data\track10.wav',r'C:\Users\Administrator\Desktop\wyspeechcode\data\across.wav']
outputflie = [r'data\track1Vad.wav',r'data\track10Vad.wav',r'data\acrossVad.wav']
def mp2Wav(source_file_path,destin_path):
    sound = AudioSegment.from_mp3(source_file_path)
    sound.export(destin_path,format ='wav')

mp2Wav(inputflie[0],outputflie[0])