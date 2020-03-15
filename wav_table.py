import struct
import numpy as np
from scipy import signal as sg
import librosa
import librosa.display
import matplotlib.pyplot as plt

# %matplotlib inline - make sure to add this your jupyter notebook session


###############################
# F R E Q E U N C Y C L A S S #
###############################

'''
This class generates a set of ocatves and overtone series from a given frequency.
It also allow you to transpose sets of frequencies. 
'''

class Hz:
    def __init__(self, hz, length=range(0,16)):
        self.hz = hz
        self.length = length

    def make_octaves(self):
        freq_list = []
        freq_list.append(self.hz)
        oct_1 = self.hz * 2
        freq_list.append(oct_1)
        oct_2 = oct_1 * 2
        freq_list.append(oct_2) 
        oct_3 = oct_2 * 2
        freq_list.append(oct_3) 
        oct_4 = oct_3 * 2 
        freq_list.append(oct_4)
        oct_5 = oct_4 * 2
        freq_list.append(oct_5)
        oct_6 = oct_5 * 2
        freq_list.append(oct_6)
        oct_7 = oct_6 * 2
        freq_list.append(oct_7)
        oct_8 = oct_7 * 2
        freq_list.append(oct_8)
        oct_9 = oct_8 * 2
        freq_list.append(oct_9)
        oct_10 = oct_9 * 2
        freq_list.append(oct_10)
        oct_11 = oct_10 * 2
        freq_list.append(oct_11)
        return freq_list


    def make_overtone_series(self):
        overtone_series = []
        for i in self.length:
            harmonic = i * self.hz
            overtone_series.append(harmonic)
        return overtone_series
    

    def transpose_hz(self, wav_list=None, transposition_amnt=None, direction=None):
        if wav_list is None:
            if direction == 'up':
                return self.hz * transposition_amnt
            if direction == 'down':
                return self.hz / transposition_amnt
        else:
            transposed_list = []
            if direction == 'up':
                for hz in wav_list:
                    transposed_list.append(hz * transposition_amnt)
            if direction == 'down':
                for hz in wav_list:
                    transposed_list.append(hz / transposition_amnt)
            else:
                print('Before transpoition can occur, please indicate what direction')
            return transposed_list

        



#############################
# S I G N A L C L A S S E S #
#############################


class Raw_Signal:
    """ 
   Raw_Signal is a class that instantiates signal objects that generate raw data files in the form of wave. 
   It does not produce a wav file. 
   
   The default is:
    sample_rate =100
    freq=3
    samples=50
    
    The default is set to this for Raw_Signal, so when turining the graph feature on, something representble shows
    """

    def __init__(self, sampling_rate=100, freq=3, samples=50, generate=None, graph=None):
        self.sampling_rate = sampling_rate
        self.freq = freq
        self.samples = samples 
        self.x = np.arange(self.samples)
        self.generate = generate
        self.graph = graph 
      
    def make_sine_wav(self):
        y = 100*np.sin(2 * np.pi * self.freq * self.x / self.sampling_rate)
        if self.generate is None:
            pass
        else:
            f = open('../raw_files/raw_{}_sine.wav'.format(self.freq), 'wb')
            for i in y:
                f.write(struct.pack('b', int(i)))
            f.close()
        if self.graph is None:
            pass
        else:
            plt.stem(self.x,y, 'r', )
            plt.plot(self.x,y)

    
    def make_square_wav(self):
        y = 100*sg.square(2 * np.pi * self.freq * self.x / self.sampling_rate)
        if self.generate is None:
            pass
        else:
            f = open('../raw_files/raw_{}_square.wav'.format(self.freq), 'wb')
            for i in y:
                f.write(struct.pack('b', int(i)))
            f.close()
        if self.graph is None:
            pass
        else:
            plt.stem(self.x,y, 'r', )
            plt.plot(self.x,y)


    def make_square_wav_duty_cycle(self, duty):
        y = 100*sg.square(2 * np.pi * self.freq * self.x / self.sampling_rate, duty)
        if self.generate is None:
            pass
        else:
            f = open('../raw_files/raw_{}_square_{}.wav'.format(self.freq, duty), 'wb')
            for i in y:
                f.write(struct.pack('b', int(i)))
            f.close()
        if self.graph is None:
            pass
        else:
            plt.stem(self.x,y,'r', )
            plt.plot(self.x,y)
    
    def make_sawtooth_wav(self):
        y = 100*sg.sawtooth(2 * np.pi * self.freq * self.x / self.sampling_rate)
        if self.generate is None:
            pass
        else:
            f = open('../raw_files/raw_{}_sawtooth.wav'.format(self.freq), 'wb')
            for i in y:
                f.write(struct.pack('b', int(i)))
            f.close()
        if self.graph is None:
            pass
        else:
            plt.stem(self.x,y, 'r', )
            plt.plot(self.x,y)

from scipy.io.wavfile import write
import wave

class Wav_Signal:

    """ 
    Wav_Signal is a  class that instantiates signal objects that generate audio waves.
    It also can make a modulated Signal and save as an audio file.

    The default is:
        sps=44100
        carrier_hz=440.0
        duration_s=10.0
        duty=0.8

    The default is set to A4. This is to allow objects to be immediately usable for example use
    """
    def __init__(self, sps=44100, carrier_hz=440.0, duration_s=10.0, duty=0.8):
        self.sps = sps
        self.carrier_hz = carrier_hz
        self.duration_s = duration_s
        self.duty = duty
        self.t_samples = np.arange(self.sps * self.duration_s)
        self.wav = 2 * np.pi * self.carrier_hz * self.t_samples / self.sps
        self.sin_carrier = np.sin(self.wav)
        self.sq_carrier = sg.square(self.wav)
        self.sq_duty_carrier = sg.square(self.wav, self.duty)
        self.saw_carrier = sg.sawtooth(self.wav)


    def make_simple_wav(self, wave):
        if wave == 'sine':
            self.sin_carrier *= 0.3
            carrier_ints = np.int16(self.sin_carrier * 32767)
            write('../wav_files/simple_{}_sin.wav'.format(self.carrier_hz), self.sps, carrier_ints)
        
        if wave == 'square':
            self.sq_carrier *= 0.3
            carrier_ints = np.int16(self.sq_carrier * 32767)
            write('../wav_files/simple_{}_square.wav'.format(self.carrier_hz), self.sps, carrier_ints)
        
        if wave == 'square_duty':
            self.sq_duty_carrier *= 0.3
            carrier_ints = np.int16(self.sq_duty_carrier * 32767)
            write('../wav_files/simple_{}_square_duty_cycle.wav'.format(self.carrier_hz), self.sps, carrier_ints)

        if wave == 'sawtooth':
            self.saw_carrier *= 0.3
            carrier_ints = np.int16(self.saw_carrier * 32767)
            write('../wav_files/simple_{}_sawtooth.wav'.format(self.carrier_hz), self.sps, carrier_ints)



    def make_mod_wav(self, wave, modulator_hz, ac, ka, graph=None, duration=None):
        if wave == 'sine':
            modulator = np.sin(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sin_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('../wav_files/mod_{}_sin.wav'.format(self.carrier_hz), self.sps, modulated_ints)
            if graph == 'graph':
                y, sr = librosa.load('mod_{}_sin.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulatd_sine')
            else:
                pass


        if wave == 'square':
            modulator = sg.square(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sq_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('../wav_files/mod_{}_sq.wav'.format(self.carrier_hz),self.sps, modulated_ints)
            if graph == 'graph':
                y, sr = librosa.load('mod_{}_sq.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulatd_square')
            else:
                pass
        
        if wave == '../wav_files/square_duty':
            modulator = sg.square(2 * np.pi * modulator_hz * self.t_samples / self.sps, self.duty)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sq_duty_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('../wav_files/mod_{}_sq_duty_cycle.wav'.format(self.carrier_hz), self.sps, modulated_ints)
            if graph == 'graph':
                y, sr = librosa.load('mod_{}_sq_duty_cycle.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulatd_square_{}_duty_cycle'.format(self.duty))
            else:
                pass

        if wave == 'sawtooth':
            modulator = sg.sawtooth(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.saw_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('../wav_files/mod_{}_saw.wav'.format(self.carrier_hz),self.sps, modulated_ints)
            if graph == 'graph':
                y, sr = librosa.load('mod_{}_saw.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulatd_sawtooth')
            else:
                pass



###################
# M E R G E W A V #
###################


'''
Functions to layer and concatentate wav files once they are generated.
'''
def stack_wav(wav_list, output_wav):
    fnames = wav_list
    wavs = [wave.open(fn) for fn in fnames]
    frames = [w.readframes(w.getnframes()) for w in wavs]
    # here's efficient numpy conversion of the raw byte buffers
    # '<i2' is a little-endian two-byte integer.
    samples = [np.frombuffer(f, dtype='<i2') for f in frames]
    samples = [samp.astype(np.float64) for samp in samples]
    # mix as much as possible
    n = min(map(len, samples))
    mix = samples[0][:n] + samples[1][:n]
    # Save the result
    mix_wav = wave.open('../wav_files/' + output_wav, 'w')
    mix_wav.setparams(wavs[0].getparams())
    # before saving, we want to convert back to '<i2' bytes:
    mix_wav.writeframes(mix.astype('<i2').tobytes())
    mix_wav.close()


def concat_wav(wav_list, output_wav):
    infiles = wav_list
    outfile = '../wav_files/' + output_wav

    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()

    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    output.writeframes(data[0][1])
    output.writeframes(data[1][1])
    output.close()
