# ----- Import Packages -----
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import IPython.display as ipd
from matplotlib.mlab import magnitude_spectrum
from pydub import AudioSegment
from pydub.playback import play
# Note: Import ffmpeg

# ----- Import Audio Files  -----
BASE_DIR = './audio'
audio_file_path = 'violin' # Change according to your audio
original_format = '.mp3'
converted_format = '.wav'

# ----- Convert MP3 files to WAV files -----
EXPORT_DIR = './converted_audio'
converted_file = AudioSegment.from_mp3(os.path.join(BASE_DIR, audio_file_path + original_format))
converted_file_path = os.path.join(EXPORT_DIR, os.path.join(audio_file_path + converted_format))
converted_file.export(converted_file_path, format='wav')
# ipd.Audio(converted_file) # Play audio on Jupyter Notebook
# play(converted_file) # Play audio on PyCharm

# ----- Loading Empire Arrays -----
audio_c4, sr = librosa.load(converted_file_path) # sr is sample_range
print(audio_c4.shape) # c4 is the signal
# Extract Fourier Transform Coefficients to have the #bins = #samples
audio_ft = np.fft.fft(audio_c4)
print(audio_ft.shape)
magnitude_spectrum_audio = np.abs(audio_ft)
print(magnitude_spectrum_audio[0])

# ----- Plotting the Magnitude Spectrum -----
def plot_magnitude_spectrum(signal, title, sr, f_ratio=1):
    ft = np.fft.fft(signal)
    mag_spectrum = np.abs(ft)

    # Calculations
    frequency = np.linspace(0, sr, len(mag_spectrum))
    num_frequency_bins = int(len(frequency) * f_ratio)

    # Plot magnitude spectrum
    plt.figure(figsize=(18, 5))
    plt.plot(frequency[:num_frequency_bins], mag_spectrum[:num_frequency_bins]) # (x, y)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)

    plt.show()

plot_magnitude_spectrum(audio_c4, "audio", sr, 0.1)

# ----- Display Graphical Overview -----