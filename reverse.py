from scipy.io.wavfile import read, write


def reverse(wav, new_wav):
    sr, y = read(wav)
    reverse_wav = y[::-1]
    write(new_wav, sr, reverse_wav)