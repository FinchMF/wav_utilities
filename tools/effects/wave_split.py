import wave
from scipy.io.wavfile import read, write

def intervals(parts, duration):
    part_duration = int(duration / parts)
    return [(i * part_duration, (i + 1) * part_duration) for i in range(int(parts))]


def divide_tracks(wav_fp, seconds_of_division):
    wav = wave.open(wav_fp, 'rb')
    frames = wav.getnframes()
    rate = wav.getframerate()
    duration = round(frames / rate, 2)
    needed_bin = duration / seconds_of_division
    factor = duration / needed_bin
    divider = intervals(needed_bin, frames)
    return [list(divider), wav, wav_fp]



def sample_divisions(divisions, wav, wav_fp, filenum, root):
    sci_wav = read(wav_fp)
    root = root
    count = 0
    for divides in divisions:
        filename = root + 'wav_{}_{}_.wav'.format(filenum, count)
        print(sci_wav[1][divides[0]:divides[1]])
        write(filename, wav.getframerate() ,sci_wav[1][divides[0]:divides[1]])
        print('rendered')
        count += 1


# takes three functions above and recieves list of wav files and seperates the wav in equal parts (by seconds)
# then saves the files in given root diretory

def split_wav_s(wav_list, num, root):
    for filenum, wav in enumerate(wav_list):
        divisions = divide_tracks(wav, num)
        sample_divisions(divisions[0],
                        divisions[1],
                        divisions[2],
                        filenum,
                        root)
    return print('Finished')