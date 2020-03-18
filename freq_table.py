

####################################
# F R E Q _ T A B L E  M O D U L E #
####################################


'''
This module contains a set of pitch and frequency tools.

pitch_to_frequency is a refrence dictionary that gives you a pitch's frequency.
This dictionary will continue to be developed to include microtonal frequencies

the pi section is a set of functions and variables that allow you to call digts of pi and transform them into
frequencies. The pi_hz_ocatve is a dictionary that gives you the octave of pi's frequences as keys. 

closet_pitch takes a frequencies and gives you the closet pitch to a given frequency.

'''

###################################
# P I T C H T O F R E Q U E N C Y #
###################################


pitch_to_frequency = {'C': {0: 16.35,
                     1: 32.70,
                     2: 65.41,
                     3: 130.81,
                     4: 261.63,
                     5: 523.25,
                     6: 1046.50,
                     7: 2093.00,
                     8: 4186.00},
               'C#': {0: 17.32,
                      1: 34.65,
                      2: 69.30,
                      3: 138.59,
                      4: 277.18,
                      5: 554.37,
                      6: 1108.73, 
                      7: 2217.46,
                      8: 4434.92},
                'D': {0: 18.35,
                      1: 36.71,
                      2: 73.42,
                      3: 146.83,
                      4: 293.66,
                      5: 587.33,
                      6: 1174.66,
                      7: 2349.32,
                      8: 4698.64},
                'D#': {0: 19.45,
                       1: 38.89,
                       2: 77.78,
                       3: 155.56,
                       4: 311.13,
                       5: 622.25,
                       6: 1244.51,
                       7: 2489.05,
                       8: 4978.03},
                'E': {0: 20.60,
                      1: 41.20,
                      2: 82.41,
                      3: 164.81,
                      4: 329.63,
                      5: 659.26,
                      6: 1318.51,
                      7: 2637.02,
                      8: 5274.04},
                'F': {0: 21.83,
                      1: 43.65,
                      2: 87.31,
                      3: 174.61,
                      4: 349.23,
                      5: 698.46,
                      6: 1396.91,
                      7: 2793.83,
                      8: 5587.65},
                'F#': {0: 23.12,
                       1: 46.25,
                       2: 92.50,
                       3: 185.00,
                       4: 369.99,
                       5: 739.99,
                       6: 1479.98,
                       7: 2959.96,
                       8: 5919.91},
                'G': {0: 24.50,
                      1: 49.00,
                      2: 98.00,
                      3: 196.00,
                      4: 392.00,
                      5: 783.99,
                      6: 1567.98,
                      7: 3135.95,
                      8: 6271.93},
                'G#': {0: 25.96,
                       1: 51.91,
                       2: 103.83,
                       3: 207.65,
                       4: 415.30,
                       5: 830.61,
                       6: 1661.22,
                       7: 3322.44,
                       8: 6644.88},
                'A': {0: 27.50,
                      1: 55.00,
                      2: 110.00,
                      3: 220.00,
                      4: 440.00,
                      5: 880.00,
                      6: 1760.00,
                      7: 3520.00,
                      8: 7040.00},
                'A#': {0: 29.14,
                       1: 58.27,
                       2: 116.54,
                       3: 233.08,
                       4: 466.16,
                       5: 932.33,
                       6: 1864.66,
                       7: 3729.31,
                       8: 7458.62},
                'B': {0: 30.87,
                      1: 61.74,
                      2: 123.47,
                      3: 246.94,
                      4: 493.88,
                      5: 987.77,
                      6: 1975.53,
                      7: 3951.07,
                      8: 7902.13}                  
            }


#######
# P I #
#######

# generate pi - the int passing through is the length of digits to call
# this function is built from the spigot algorithim
def make_pi(int):
    q, r, t, k, m, x = 1, 0, 1, 1, 3, 3
    for j in range(int):
        if 4 * q + r - t < m * t:
            yield m
            q, r, t, k, m, x = 10*q, 10*(r-m*t), t, k, (10*(3*q+r))//t - 10*m, x
        else:
            q, r, t, k, m, x = q*k, (2*q+r)*x, t*x, k+1, (q*(7*k+2)+r*x)//(t*x), x+2


# generate print statment of digits called from make_pi algorhithm
def digits_of_pi(int):
    digits_of_pi = []
    for i in make_pi(int):
        digits_of_pi.append(str(i))
    pi_digits = digits_of_pi[:1] + ['.'] + digits_of_pi[1:]
    pi_string = "".join(pi_digits)
    return print(str(len(pi_string) - 2 ) + ' digits of pi:\n %s' % pi_string)

#--------------------------------------------------------------------------------------------------

# pi_call to make 1157 digits of pi
digit_of_pi = []
for i in make_pi(5000):
      digit_of_pi.append(str(i))

pi_digits = digit_of_pi[:1] + ['.'] + digit_of_pi[1:]
pi_string = "".join(pi_digits)
# grab the 3.14 as float to make frequency
pi = float(pi_string[:4])

#--------------------------------------------------------------------------------------------------

# make octaves for pi
def make_octaves(freq):
     freq_list = []
     freq_list.append(freq)
     oct_1 = freq * 2
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

# call octaves
pi_octaves = make_octaves(pi)
# call octave range
octaves = range(-3,9)
# pull pi octave dictionary dictionary 
pi_hz_octave = dict(zip(octaves, pi_octaves))

#################################################
# D E T E C T  N E I G H B O R I N G  P I T C H #
#################################################

# use closest pitch to calulate nearest pitch to passed frequency
# if pitch is exact, then it can be used to tell you the name of the pitch
from math import log2, pow

A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
def closest_pitch(freq):
    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)
