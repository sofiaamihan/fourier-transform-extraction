# ----- Import Packages -----
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from pydub import AudioSegment
from pydub.playback import play
# Note: Import ffmpeg
import matplotlib.animation as animation
import threading

# ----- Import Audio Files  -----
BASE_DIR = './audio'
audio_file_path = 'all_i_ask_palmer' # Change Accordingly
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
def plot_magnitude_spectrum(signal, sr, f_ratio=1):
    ft = np.fft.fft(signal)
    mag_spectrum = np.abs(ft)

    # Calculations
    frequency = np.linspace(0, sr, len(mag_spectrum))
    num_frequency_bins = int(len(frequency) * f_ratio)

    # Plot magnitude spectrum
    plt.figure(figsize=(12, 5))
    plt.plot(frequency[:num_frequency_bins], mag_spectrum[:num_frequency_bins], 'pink') # (x, y)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Magnitude Spectrum")

    plt.show()

# plot_magnitude_spectrum(audio_c4, sr, 0.1)

# ----- Plotting the Animated Magnitude Spectrum -----
def plot_animated_magnitude_spectrum(signal, sr, window_size=1024, f_ratio=1):
    num_frames = len(signal) // window_size
    fig, ax = plt.subplots(figsize=(18, 5))

    # Calculations
    frequency = np.linspace(0, sr, window_size)
    num_frequency_bins = int(len(frequency) * f_ratio)

    # Plot magnitude spectrum
    line, = ax.plot([], [], 'pink')
    ax.set_xlim(0, frequency[num_frequency_bins-1])
    ax.set_ylim(0, 200)  # Adjust height accordingly
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Animated Magnitude Spectrum")

    # Initialise function to compute FFT and update plot for each frame
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        start_idx = frame * window_size
        end_idx = (frame + 1) * window_size
        window_signal = signal[start_idx:end_idx]

        # Compute FFT and magnitude spectrum for this frame
        ft = np.fft.fft(window_signal)
        mag_spectrum = np.abs(ft)

        # Update the plot
        line.set_data(frequency[:num_frequency_bins], mag_spectrum[:num_frequency_bins])
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=50)

    def play_audio():
        play(converted_file)

    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

    plt.show()

plot_animated_magnitude_spectrum(audio_c4, sr, window_size=1024, f_ratio=0.1)
