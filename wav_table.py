import struct
import numpy as np
from scipy import signal as sg



class Single_wav:
    def __init__(self, sampling_rate, freq, samples):
        self.sampling_rate = sampling_rate
        self.freq = freq
        self.samples = samples 
        self.x = np.arange(self.samples)

        

    def make_sine_wav(self):
        y = 100*np.sin(2 * np.pi * self.freq * self.x / self.sampling_rate)
        f = open('single_sine.wav', 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()
    
    def make_square_wav(self):
        y = 100*sg.square(2 * np.pi * self.freq * self.x / self.sampling_rate)
        f = open('single_square.wav', 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()

    def make_square_wav_duty_cycle(self, duty):
        y = 100*sg.square(2 * np.pi * self.freq * self.x / self.sampling_rate, duty)
        f = open('single_square_{}.wav'.format(duty), 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()
    
    def make_sawtooth_wav(self):
        y = 100*sg.sawtooth(2 * np.pi * self.freq * self.x / self.sampling_rate)
        f = open('single_sawtooth.wav', 'wb')
        for i in y:
            f.write(struct.pack('b', int(i)))
        f.close()

from scipy.io.wavfile import write

class Signal_Modulation:
    def __init__(self, sps, carrier_hz, duration_s):
        self.sps = sps
        self.carrier_hz = carrier_hz
        self.duration_s = duration_s
        self.t_samples = np.arange(self.sps * self.duration_s)
        self.carrier = np.sin(2 * np.pi * self.carrier_hz * self.t_samples / self.sps)
        
    def make_simple_sine_wav(self):
        self.carrier *= 0.3
        carrier_ints = np.int16(self.carrier * 32767)
        write('simple_sin.wav', self.sps, carrier_ints)

    def modulate_the_carrier(self, modulator_hz, ac, ka):
        modulator = np.sin(2 * np.pi * modulator_hz * self.t_samples / self.sps)
        envelope = ac * (1.0 + ka * modulator)
        modulated = envelope * self.carrier
        modulated *= 0.3
        modulated_ints = np.int16(modulated * 32767)
        write('amplitude_modulated.wav', self.sps, modulated_ints)
