# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import IPython.display as ipd
from pydub import AudioSegment
from pydub.playback import play
# Note: Import ffmpeg

# Import Audio Files / Ask User for Audio File
BASE_DIR = './audio'
audio_file = 'all_i_ask_palmer.mp3'

# Convert MP3 files to WAV files
converted_file = AudioSegment.from_mp3(os.path.join(BASE_DIR, audio_file))
# ipd.Audio(converted_file) # This is for jupyter notebook
play(converted_file)

# Loading Empire Arrays
# Plotting the Magnitude Spectrum
# Display Graphical Overview