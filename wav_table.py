import struct
import numpy as np
from scipy import signal as sg



class Raw_Signal:
    """ 
   Raw_Signal is a class that instantiates signal objects that generate raw data files in the form of wave. 
   It does not produce a wav file.
    """

    def __init__(self, sampling_rate, freq, samples):
        self.sampling_rate = sampling_rate
        self.freq = freq
        self.samples = samples 
        self.x = np.arange(self.samples)
      
    def make_sine_wav(self):
        y = 100*np.sin(2 * np.pi * self.freq * self.x / self.sampling_rate)
        f = open('raw_sine.wav', 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()
    
    def make_square_wav(self):
        y = 100*sg.square(2 * np.pi * self.freq * self.x / self.sampling_rate)
        f = open('raw_square.wav', 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()

    def make_square_wav_duty_cycle(self, duty):
        y = 100*sg.square(2 * np.pi * self.freq * self.x / self.sampling_rate, duty)
        f = open('raw_square_{}.wav'.format(duty), 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()
    
    def make_sawtooth_wav(self):
        y = 100*sg.sawtooth(2 * np.pi * self.freq * self.x / self.sampling_rate)
        f = open('raw_sawtooth.wav', 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()

from scipy.io.wavfile import write

class Wav_Signal:

    """ 
    Wav_Signal is a  class that instantiates signal objects that generate audio waves.
    It also can make a modulated Signal and save as an audio file.
    """
    def __init__(self, sps, carrier_hz, duration_s):
        self.sps = sps
        self.carrier_hz = carrier_hz
        self.duration_s = duration_s
        self.t_samples = np.arange(self.sps * self.duration_s)
        self.wav = 2 * np.pi * self.carrier_hz * self.t_samples / self.sps
        self.sin_carrier = np.sin(self.wav)
        self.sq_carrier = sg.square(self.wav)
        self.saw_carrier = sg.sawtooth(self.wav)

    def make_simple_sine_wav(self):
        self.sin_carrier *= 0.3
        carrier_ints = np.int16(self.sin_carrier * 32767)
        write('simple_sin.wav', self.sps, carrier_ints)

    def make_simple_square_wav(self):
        self.sq_carrier *= 0.3
        carrier_ints = np.int16(self.sq_carrier * 32767)
        write('simple_square.wav', self.sps, carrier_ints)
    
    def make_simple_sawtooth_wav(self):
        self.saw_carrier *= 0.3
        carrier_ints = np.int16(self.saw_carrier * 32767)
        write('simple_sawtooth.wav', self.sps, carrier_ints)

    def make_mod_wav(self, wave, modulator_hz, ac, ka):
        if wave == 'sine':
            modulator = np.sin(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sin_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('mod_sin.wav', self.sps, modulated_ints)

        if wave == 'square':
            modulator = sg.square(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.sq_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('mod_sq.wav',self.sps, modulated_ints)

        if wave == 'sawtooth':
            modulator = sg.sawtooth(2 * np.pi * modulator_hz * self.t_samples / self.sps)
            envelope = ac * (1.0 + ka * modulator)
            modulated = envelope * self.saw_carrier
            modulated *= 0.3
            modulated_ints = np.int16(modulated * 32767)
            write('mod_saw.wav',self.sps, modulated_ints)