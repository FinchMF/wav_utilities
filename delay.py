import wave
from audioop import add, mul
from warnings import warn

def input_wave(filename,frames=10000000): #10000000 is an arbitrary large number of frames
    with wave.open(filename,'rb') as wave_file:
        params=wave_file.getparams()
        audio=wave_file.readframes(frames)  
        if params.nchannels!=1:
            raise Exception("The input audio should be mono for these examples")
    return params, audio

#output to file so we can use ipython notebook's Audio widget
def output_wave(audio, params, stem, suffix):
    #dynamically format the filename by passing in data
    filename=stem.replace('.wav','_{}.wav'.format(suffix))
    with wave.open(filename,'wb') as wf:
        wf.setparams(params)
        wf.writeframes(audio)
        wf.close()


def delay(audio_bytes,params,offset_ms):
    """version 1: delay after 'offset_ms' milliseconds"""
    #calculate the number of bytes which corresponds to the offset in milliseconds
    offset= params.sampwidth*offset_ms*int(params.framerate/1000)
    #create some silence
    beginning= b'\0'*offset
    #remove space from the end
    end= audio_bytes[:-offset]
    return add(audio_bytes, beginning+end, params.sampwidth)




#new delay function with factor
def delay_2(audio_bytes,params,offset_ms,factor=1):
    """version 2: delay after 'offset_ms' milliseconds amplified by 'factor'"""
    #calculate the number of bytes which corresponds to the offset in milliseconds
    offset= params.sampwidth*offset_ms*int(params.framerate/1000)
    #create some silence
    beginning= b'\0'*offset
    #remove space from the end
    end= audio_bytes[:-offset]
    #multiply by the factor
    multiplied_end= mul(audio_bytes[:-offset],params.sampwidth,factor)
    return add(audio_bytes, beginning+ multiplied_end, params.sampwidth)


def delay_3(audio_bytes,params,offset_ms,factor=1,num=1):
    """version 3: 'num' delays after 'offset_ms' milliseconds amplified by 'factor'"""
    if factor>=1:
        warn("These settings may produce a very loud audio file. \
             Please use caution when listening")
    #calculate the number of bytes which corresponds to the offset in milliseconds
    offset=params.sampwidth*offset_ms*int(params.framerate/1000)
    #add extra space at the end for the delays
    delayed_bytes=audio_bytes
    for i in range(num):
        #create some silence
        beginning = b'\0'*offset*(i+1)
        #remove space from the end
        end = audio_bytes[:-offset*(i+1)]
        #multiply by the factor
        multiplied_end= mul(end,params.sampwidth,factor**(i+1))
        delayed_bytes= add(delayed_bytes, beginning+multiplied_end, params.sampwidth)
    return delayed_bytes


def delay_4(audio_bytes,params,offset_ms,factor=1,num=1):
    """version 4: 'num' delays after 'offset_ms' milliseconds amplified by 'factor' 
    with additional space"""
    if factor>=1:
        warn("These settings may produce a very loud audio file. \
              Please use caution when listening")
    #calculate the number of bytes which corresponds to the offset in milliseconds
    offset=params.sampwidth*offset_ms*int(params.framerate/1000)
    #add extra space at the end for the delays
    audio_bytes=audio_bytes+b'\0'*offset*(num)
    #create a copy of the original to apply the delays
    delayed_bytes=audio_bytes
    for i in range(num):
        #create some silence
        beginning = b'\0'*offset*(i+1)
        #remove space from the end
        end = audio_bytes[:-offset*(i+1)]
        #multiply by the factor
        multiplied_end= mul(end,params.sampwidth,factor**(i+1))
        delayed_bytes= add(delayed_bytes, beginning+multiplied_end, params.sampwidth)
    return delayed_bytes






