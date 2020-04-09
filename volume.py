import wave, numpy, struct

def adjust_volume(wav, new_wav, adjust, factor):
 
    # Open
    w = wave.open(wav,"rb")
    p = w.getparams()
    f = p[3] # number of frames
    s = w.readframes(f)
    w.close()

    # Edit
    if adjust == 'decrease':
        s = numpy.fromstring(s, numpy.int16) // factor  # half amplitude
        s = struct.pack('h'*len(s), *s)
    if adjust == 'increase':
        s = numpy.fromstring(s, numpy.int16) * factor  # half amplitude
        s = struct.pack('h'*len(s), *s)


    # Save
    w = wave.open(new_wav,"wb")
    w.setparams(p)
    w.writeframes(s)
    w.close()




