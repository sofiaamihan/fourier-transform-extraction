# Fourier Transform Extraction
A lightweight Python tool to visualise and analyse the frequency content of audio signals using the Fast Fourier Transform (FFT). 
This project displays both static and animated magnitude spectrum visualisations along with real-time audio playback.

![Example Image]()

## Features
- **Static Spectrum Plot**: Displays a single-frame magnitude spectrum of the full signal.
- **Animated Spectrum Plot**: Visualises frequency evolution over time, synchronised with its respective audio playback.
- **MP3 to WAV Conversion**: Automated audio processing.

## Packages Used
- `numpy`
- `matplotlib`
- `os`
- `librosa`
- `pydub`
- `ffmpeg` (dependency for audio conversion/playback)
- `threading` (for simultaneous playback and plotting)

> Ensure that `ffmpeg` is installed and accessible in your system's PATH.

## Setup Instructions
1. Clone Repository
2. Install Python Dependencies
3. Install `ffmpeg`
4. Place your MP3 File in the `./audio` folder
5. Edit `audio_file_path` in **line 14** accordingly.
