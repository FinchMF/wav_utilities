
import struct
import numpy as np
from scipy import signal as sg

# Here are four functions to generate single tone wav files. 
# Task: build Wav Class that construct them

###################
# S I N E W A V E #
###################

sampling_rate = 44100                    ## Sampling Rate
freq = 440                               ## Frequency (in Hz)
samples = 44100                          ## Number of samples 
x = np.arange(samples)

####### sine wave ###########
y = 100*np.sin(2 * np.pi * freq * x / sampling_rate)


f = open('sin_test.wav','wb')

for i in y:
	f.write(struct.pack('b',int(i)))
f.close()


#######################
# S Q U A R E W A V E #
#######################


sampling_rate = 44100                    ## Sampling Rate
freq = 440                               ## Frequency (in Hz)
samples = 44100                          ## Number of samples 
x = np.arange(samples)

####### square wave ##########
y = 100* sg.square(2 *np.pi * freq * x / sampling_rate)

f = open('square_test.wav','wb')

for i in y:
	f.write(struct.pack('b',int(i)))
f.close()

##############################################
# S Q U A R E W A V E with D U T Y C Y C L E #
##############################################

sampling_rate = 44100                    ## Sampling Rate
freq = 440                               ## Frequency (in Hz)
samples = 44100                          ## Number of samples 
x = np.arange(samples)

####### square wave with Duty Cycle ##########
y = 100* sg.square(2 *np.pi * freq * x / sampling_rate , duty = 0.8)


f = open('square_2_test.wav','wb')

for i in y:
	f.write(struct.pack('b',int(i)))
f.close()


###########################
# S A W T O O T H W A V E #
###########################


sampling_rate = 44100                    ## Sampling Rate
freq = 440                               ## Frequency (in Hz)
samples = 44100                          ## Number of samples 
x = np.arange(samples)

####### Sawtooth wave ########
y = 100* sg.sawtooth(2 *np.pi * freq * x / sampling_rate)


f = open('sawtooth_test.wav','wb')

for i in y:
	f.write(struct.pack('b',int(i)))
f.close()



# Something to consider: another method to make a single tone wave. along with it is a way to work with modulation and amplitude.
# Task: incorporate this into the Class



##################
# Another Method #
##################

#Another method to generate a simple sine wave and then make

import numpy as np
from scipy.io.wavfile import write

# Properties of the wav
sps = 44100
carrier_hz = 440.0
duration_s = 10.0

# Calculate the sine wave
t_samples = np.arange(sps * duration_s)
carrier = np.sin(2 * np.pi * carrier_hz * t_samples / sps)
carrier *= 0.3
carrier_ints = np.int16(carrier * 32767)

# Write the wav file
write('simple-sine.wav', sps, carrier_ints)



# Amplitude and Modulation
# Properties of the wav
sps = 44100    # DON'T change

carrier_hz = 440.0
modulator_hz = 0.25
ac = 1.0
ka = 0.25
duration_s = 10.0

# Calculate the sine wave
t_samples = np.arange(sps * duration_s)
carrier = np.sin(2 * np.pi * carrier_hz * t_samples / sps)

# Modulate the carrier
modulator = np.sin(2 * np.pi * modulator_hz * t_samples / sps)
envelope = ac * (1.0 + ka * modulator)
modulated = envelope * carrier


# Write the wav file
modulated *= 0.3
modulated_ints = np.int16(modulated * 32767)
write('amplitude-modulated.wav', sps, modulated_ints)


###############################
# H A R M O N I C S E R I E S #
###############################

# Debug this. Make multiple pitches in a row that out line the harmonic series. Add to Class


import struct
import numpy as np
from scipy import signal as sg
from scipy.io.wavfile import write

sampling_rate = 44100                    ## Sampling Rate
freq = 440                               ## Frequency (in Hz)
samples = 44100                          ## Number of samples 
x = np.arange(samples)

####### sine wave ###########
y = 100*np.sin(2 * np.pi * freq * x / sampling_rate)



octave = []
for i in y: 
    f2 = 2*i
    octave.append(f2)

fifth_above_octave = []
for i in y:
    f3 = 3*i
    fifth_above_octave.append(f3)

fourth_above_fifth = []
for i in y:
    f4 = 4*i
    fourth_above_fifth.append(f4)

f1 = list(y)
f2 = list(octave)
f3 = list(fifth_above_octave)
f4 = list(fourth_above_fifth)

harmonic = f1 + f2 + f3 + f4

hny = np.array(harmonic)



f = open('harm_sin_test.wav','wb')

for i in hny:
	f.write(struct.pack('b',int(i)))
f.close()


##############
# Next Steps #
##############

# Once this Class is finished, the next step will be to use OS and have the script open a DAW with the new file in it. 
# This is a stretch goal but would be fun. 

