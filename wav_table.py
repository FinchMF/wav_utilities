

##################################
# W A V _ T A B L E  M O D U L E #
##################################

import struct
import numpy as np
from scipy import signal as sg
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
import glob

# %matplotlib inline - make sure to add this to your notebook session


# object property inspector
def inspect(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))


interval_systems = {'semi_tones': 2**(1/12),
                    'tri_tones': 2**(1/18),
                    'quarter_tones': 2**(1/24),
                    'sixth_tones': 2**(1/36),   
                    'eigth_tones': 2**(1/48),
                    'twelevth_tones': 2**(1/72),
                    'sixteenth_tones': 2**(1/96),
                    'twenty_fourth_tones': 2**(1/144)}

###############################
# F R E Q E U N C Y C L A S S #
###############################

'''
This class generates a set of ocatves and overtone series from a given frequency.

The length is default 0 - 16. This is to allow the overtone series to
be 17 partials. 

It also can transpose sets of frequencies. 
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

    def make_system(self, system_type=interval_systems.get('semi_tones'), system_size=range(0,12)):
        system = {}
        freq_position = 0
        for p in system_size:
            freq = self.hz * (system_type)**p
            system['freq_{}'.format(freq_position)] = round(freq, 2)
            freq_position += 1
        return system
    
    '''
    The parameters for transpose_hz allow for a variety of transpositon types. 
    If 'down octave' is chosen, the transposition amnt needs to be a negative int.
    If no list is passed, the function will default to transposing the frequency of the Hz object.
    '''

    def transpose_hz(self, wav_list=None, transposition_amnt=None, direction=None):
        if wav_list is None:
            if direction == 'up cents':
                return round(self.hz *2**(round(transposition_amnt,2)/1200),2)
            if direction == 'up series':
                return self.hz * transposition_amnt
            if direction == 'up octave':
                return self.hz *2**transposition_amnt
            if direction == 'down cents': 
                return round(self.hz *2**(round((-1*transposition_amnt),2)/1200),2)
            if direction == 'down series':
                return self.hz / transposition_amnt
            if direction == 'down octaves':
                return self.hz *2**(-1*transposition_amnt)
        else:
            if direction is None:
                print('Before transpoition can occur, please indicate what direction')
            else:
                transposed_list = []
                if direction == 'up cents':
                    for hz in wav_list:
                        transposed_list.append(round(hz*2**(round(transposition_amnt,2)/1200),2))
                if direction == 'up series':
                    for hz in wav_list:
                        transposed_list.append(hz * transposition_amnt)
                if direction == 'up octaves':
                    for hz in wav_list:
                        transposed_list.append(hz *2**transposition_amnt)
                if direction == 'down cents':
                    for hz in wav_list:
                        transposed_list.append(round(hz*2**(round((-1*transposition_amnt),2)/1200),))        
                if direction == 'down series':
                    for hz in wav_list:
                        transposed_list.append(hz / transposition_amnt)
                if direction == 'down octaves':
                    for hz in wav_list:
                        transposed_list.append(hz *2**(-1*transposition_amnt))
            
        return transposed_list

#############################
# S I G N A L C L A S S E S #
#############################

class Raw_Signal:
    """ 
   Raw_Signal is a class that instantiates signal objects that generate raw data files in the form of a wave. 
   When generate is set to on, it does not produce a wav file. However, sing audacity, you are able
   to convert the file into an audio file.
   
   The default is:
        sample_rate=100
        freq=3
        samples=50
        generate=None
        graph=None
    
    The default is set to these parameters to allow the graph feature to have a coherant representation when turned on.
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
            f = open('raw_files/raw_{}_sine.wav'.format(self.freq), 'wb')
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
            f = open('raw_files/raw_{}_square.wav'.format(self.freq), 'wb')
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
            f = open('raw_files/raw_{}_square_{}.wav'.format(self.freq, duty), 'wb')
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
            f = open('raw_files/raw_{}_sawtooth.wav'.format(self.freq), 'wb')
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
    Wav_Signal is a class that instantiates signal objects that generate audio waves.
    It can make a modulated Signal and save as an audio file as well.

    The default is:
        sps=44100
        carrier_hz=440.0
        duration_s=10.0
        duty=0.8

    The default is set to A4. This is to allow objects to be immediately usable for as an example.
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
            write('wav_files/simple_{}_sin.wav'.format(self.carrier_hz), self.sps, carrier_ints)
        
        if wave == 'square':
            self.sq_carrier *= 0.3
            carrier_ints = np.int16(self.sq_carrier * 32767)
            write('wav_files/simple_{}_square.wav'.format(self.carrier_hz), self.sps, carrier_ints)
        
        if wave == 'square_duty':
            self.sq_duty_carrier *= 0.3
            carrier_ints = np.int16(self.sq_duty_carrier * 32767)
            write('wav_files/simple_{}_square_duty_cycle.wav'.format(self.carrier_hz), self.sps, carrier_ints)

        if wave == 'sawtooth':
            self.saw_carrier *= 0.3
            carrier_ints = np.int16(self.saw_carrier * 32767)
            write('wav_files/simple_{}_sawtooth.wav'.format(self.carrier_hz), self.sps, carrier_ints)

    '''
    Modulating the wave's amplitude is set at these defaults:

    modulator_hz=0.25
    ac=1.0
    ka=0.25
    graph=None (this means graph is off)
    duration=None (this refers to the duration of the graph's window)
    '''

    def make_mod_wav(self, wave, modulator_hz=0.25, ac=1.0, ka=0.25, graph=None, duration=None):
        if wave == 'sine':
            modulator = np.sin(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sin_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('wav_files/mod_{}_sin.wav'.format(self.carrier_hz), self.sps, modulated_ints)
            if graph == 'on':
                y, sr = librosa.load('wav_files/mod_{}_sin.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulated_sine')
            else:
                pass

        if wave == 'square':
            modulator = sg.square(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sq_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('wav_files/mod_{}_sq.wav'.format(self.carrier_hz),self.sps, modulated_ints)
            if graph == 'on':
                y, sr = librosa.load('wav_files/mod_{}_sq.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulated_square')
            else:
                pass
        
        if wave == 'square_duty':
            modulator = sg.square(2 * np.pi * modulator_hz * self.t_samples / self.sps, self.duty)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sq_duty_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('wav_files/mod_{}_sq_duty_cycle.wav'.format(self.carrier_hz), self.sps, modulated_ints)
            if graph == 'on':
                y, sr = librosa.load('wav_files/mod_{}_sq_duty_cycle.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulated_square_{}_duty_cycle'.format(self.duty))
            else:
                pass

        if wave == 'sawtooth':
            modulator = sg.sawtooth(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.saw_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('wav_files/mod_{}_saw.wav'.format(self.carrier_hz),self.sps, modulated_ints)
            if graph == 'on':
                y, sr = librosa.load('wav_files/mod_{}_saw.wav'.format(self.carrier_hz), duration=duration)
                plt.figure()
                plt.subplot(3, 1, 1)
                librosa.display.waveplot(y, sr=sr)
                plt.title('modulated_sawtooth')
            else:
                pass


###################
# M E R G E W A V #
###################

'''
Functions to layer and concatentate wav files once they are generated.

stack_wav can only layer two waves at a time. If you end up with a list of waves
you'd like to stack, callable_stacked_pairs will take the the list of files and 
convert them into a list of tuples that have the coupled waves that will be stacked
once the list is iterated through. 

it is important to note that the wave list must be an equal number of waves. 

in order to streamline the the functionality given a larger list, the process is as follows:

Here is an example with 128 files. Which turns to 64 tuples.

assume pi_wavs is a the list of 128 files
the first line reduces twice. 128 - > 64 -> 32

1. stack_quad = stack_inital(pi_wavs, 'stack_pair', 'stack_quad')
- 32 remain
2. stack_layers('wav_files/'+stack_quad+'*.wav', 'stack_oct')
- 16 reamin 
3. stack_layers('wav_files/stack_oct*.wav', 'stack_16')
- 8 remain
4. stack_layers('wav_files/stack_16*.wav', 'stack_32')
- 4 remain
5. stack_layers('wav_files/stack_32*.wav', 'stack_64')
- 2 remain
6. s_list = glob.glob('wav_files/stack_64*.wav')
  stack_wav(s_list, 'Final_PI_stack.wav')

line 6 takes the remaining 2 files, groups them into a list and passes it
through stack_wav and titles the output file. The same process can be used with 
Concatenation of the files. 


*** will be be building a function that counts the files ina list and outputs
whether it there is an even or odd amount of files and how may layers will be
necessary to reduce down to a final file ***

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
    mix_wav = wave.open('wav_files/' + output_wav, 'w')
    mix_wav.setparams(wavs[0].getparams())
    # before saving, we want to convert back to '<i2' bytes:
    mix_wav.writeframes(mix.astype('<i2').tobytes())
    mix_wav.close()


def callable_stacked_pairs(pairs, filename):
    count = 0
    for pair in pairs:
        fname = filename + '_{}_.wav'.format(count)
        stack_wav(pair, fname)
        count += 1
    return print('All callable pairs are titled with: ' + filename)

#---------------------------------------------------------------------------------------

def stack_inital(wav_list, filename, fname):
    pair_list = list()
    while(wav_list):
        a = wav_list.pop(0); b = wav_list.pop(0)
        pair_list.append((a,b))
    callable_stacked_pairs(pair_list, filename)
    old_fname = 'wav_files/'+filename +'*.wav' 
    stack_layers(old_fname, fname)
    return fname


def stack_layers(str, fname):
    list_of_wavs = glob.glob(str)
    list_of_tupled_wavs = list()
    while(list_of_wavs):
        a = list_of_wavs.pop(0); b =list_of_wavs.pop(0)
        list_of_tupled_wavs.append((a,b))
    count = 0
    for paired in list_of_tupled_wavs:
        name = fname + '_{}.wav'.format(count)
        stack_wav(paired,name)
        count += 1
    return print('All coupled files are now stacked')

#---------------------------------------------------------------------------------------

def concat_wav(wav_list, output_wav):
    infiles = wav_list
    outfile = 'wav_files/' + output_wav
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


def callable_concated_pairs(pairs, filename):
    count = 0
    for pair in pairs:
        fname = filename + '_{}_.wav'.format(count)
        concat_wav(pair, fname)
        count += 1
    return print('All callable pairs are titled with: ' + filename)    

#---------------------------------------------------------------------------------------

def concat_inital(wav_list, filename, fname):
    pair_list = list()
    while(wav_list):
        a = wav_list.pop(0); b = wav_list.pop(0)
        pair_list.append((a,b))
    callable_concated_pairs(pair_list, filename)
    old_fname = 'wav_files/'+filename +'*.wav' 
    concat_layers(old_fname, fname)
    return fname


def concat_layers(str, fname):
    list_of_wavs = glob.glob(str)
    list_of_tupled_wavs = list()
    while(list_of_wavs):
        a = list_of_wavs.pop(0); b =list_of_wavs.pop(0)
        list_of_tupled_wavs.append((a,b))
    count = 0
    for paired in list_of_tupled_wavs:
        name = fname + '_{}.wav'.format(count)
        concat_wav(paired,name)
        count += 1
    return print('All coupled files are now connected')

#######################
# S H O W S I G N A L #
#######################

''' Show signal plots the wav file in a spectrogram'''

def show_signal(rec):
    y, sr = librosa.load(rec)

    # chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                                ref=np.max)

    plt.figure(figsize=(24, 8))
    librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
                            bins_per_octave=BINS_PER_OCTAVE,
                            x_axis='time')
    return plt.tight_layout()

########################################
# S T R E T C H  A L G O R I T H I M S #
########################################

def stretch_algorithim_1(infile, factor, outfile):
    CHANNELS = 1
    swidth = 2
    # slow down and speed up wav files 
    # if the number is under 1 it is slower, if the number is above 1, it is faster
    factor = factor
    spf = wave.open(infile, 'rb')
    RATE=spf.getframerate()
    signal = spf.readframes(-1)
    wf = wave.open(outfile, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(swidth)
    wf.setframerate(RATE*factor)
    wf.writeframes(signal)
    wf.close()
    print('stretch_algorithim_1 complete')

def stretch_algorithim_2(infile, factor, outfile):
    # to time stretch recording (pitch is not as affected)
    # less that 1 speeds up the audio, greater than 1 slows down the audio
    infile=wave.open(infile, 'rb')
    rate= infile.getframerate()
    channels=infile.getnchannels()
    swidth=infile.getsampwidth()
    nframes= infile.getnframes()
    audio_signal= infile.readframes(nframes)
    outfile = wave.open(outfile, 'wb')
    outfile.setnchannels(channels)
    outfile.setsampwidth(swidth)
    outfile.setframerate(rate/factor)
    outfile.writeframes(audio_signal)
    outfile.close()
    return print('stretch_algorithim_2 complete')

def stretch_algorithim_3(infile, factor, outfile):
    y, sr = librosa.load(infile)
    time_shift = librosa.effects.time_stretch(y, factor)
    librosa.output.write_wav(outfile, time_shift, sr)
    return print('strech_algorithim_3 complete') 


############################################
# P I T C H S H I F T  A L G O R I T H I M #
############################################

def pitch_shift_1(infile, outfile, n_steps, base_of_octave_divison):
    y, sr = librosa.load(infile)
    pitch_shift = librosa.effects.pitch_shift(y, sr, 
                                              n_steps=n_steps, 
                                              bins_per_octave=base_of_octave_divison)
    librosa.output.write_wav(outfile, pitch_shift, sr)
    return print('pitch_shift_1 complete')

def pitch_shift_2(infile, outfile, hz):
    # this function isn't that great
    # pitch up
    wr = wave.open(infile, 'r')
    # Set the parameters for the output file.
    par = list(wr.getparams())
    par[3] = 0  # The number of samples will be set by writeframes.
    par = tuple(par)
    ww = wave.open(outfile, 'w')
    ww.setparams(par)
    fr = 20
    sz = wr.getframerate()//fr  # Read and process 1/fr second at a time.
    # A larger number for fr means less reverb.
    c = int(wr.getnframes()/sz)  # count of the whole file
    shift = hz//fr 
    for _ in range(c):
        da = np.fromstring(wr.readframes(sz), dtype=np.int16)
        left, right = da[0::2], da[1::2]  # left and right channel
        lf, rf = np.fft.rfft(left), np.fft.rfft(right)
        lf, rf = np.roll(lf, shift), np.roll(rf, shift)
        lf[0:shift], rf[0:shift] = 0, 0
        nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
        ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
        ww.writeframes(ns.tostring())
    wr.close()
    ww.close()
    return print('pitch_shift_2 complete')


##########################
# D E T E C T  P I T C H #
##########################

def detech_pitch(fname):

    data_size = 40000
    fname = fname
    wav_file = wave.open(fname, 'r')
    frate = wav_file.getframerate()
    data = wav_file.readframes(data_size)
    wav_file.close()
    data = struct.unpack('{n}h'.format(n=data_size), data)
    data = np.array(data)

    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w))
    print(freqs.min(), freqs.max())
    # (-0.5, 0.499975)

    # Find the peak in the coefficients
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * frate)
    print(freq_in_hertz)
    return freq_in_hertz



#########################
# S A V E O B J E C T S #
#########################

'''
these function are for saving dictionaries of wav classes.
although they can be used to save anytime of object.
'''

def save_obj(obj, filename):
    with open(filename, 'wb') as saved_obj:
        pickle.dump(obj, saved_obj)


def load_obj(filename):
    with open(filename, 'rb') as loaded_obj:
        new_obj = pickle.load(loaded_obj)
    return new_obj