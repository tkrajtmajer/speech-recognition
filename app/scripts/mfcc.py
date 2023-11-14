import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_audio(audio):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio, sr=sr, color='b')
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

def plot_mfccs(mfccs):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title('MFCCs')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    plt.show()

def load_audio_and_mfcc(file_path, show_plots=False):
    """Computes MFCC coefficients from audio at specified path

    Args:
        file_path (string): audio path
        show_plots (bool, optional): Audio and MFCCs can be plotted for visual inspection. Defaults to False.

    Returns:
        mfccs: computed MFCCS
    """
    audio, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr) # n_mfcc param excluded

    # Optionally audio and corresponding MFCCs can be plotted for inspection
    if show_plots:
        plot_audio(audio)
        plot_mfccs(mfccs)

    return mfccs
