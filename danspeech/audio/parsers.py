from abc import ABC, abstractmethod

import torch

import scipy
import numpy as np
import librosa

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class AudioParser(ABC):
    """
    Abstract class for AudioParsers
    """

    def __init__(self, audio_config=None):
        self.audio_config = audio_config
        if not self.audio_config:
            self.audio_config = {}

        # Defaulting if no audio_config
        self.normalize = self.audio_config.get("normalize", True)
        self.sampling_rate = self.audio_config.get("sampling_rate", 16000)
        window = self.audio_config.get("window", "hamming")
        self.window = windows[window]

        self.window_stride = self.audio_config.get("window_stride", 0.01)
        self.window_size = self.audio_config.get("window_size", 0.02)

    @abstractmethod
    def parse_audio(self, recording):
        pass


class SpectrogramAudioParser(AudioParser):
    """
    Class for Spectrogram parsing

    """

    def __init__(self, audio_config=None):
        # inits all audio configs
        super(SpectrogramAudioParser, self).__init__(audio_config)

        self.n_fft = int(self.sampling_rate * self.window_size)
        self.hop_length = int(self.sampling_rate * self.window_stride)

    def parse_audio(self, recording):
        """
        Parses the given recording to a spectrogram for DanSpeech models.

        :param recording: Audio/Speech data in numpy array format.
        :return: Spectrogram
        """

        # STFT
        D = librosa.stft(recording, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window=self.window)

        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect


class InferenceSpectrogramAudioParser(AudioParser):
    """
    Class for Adaptive Spectrogram parsing.

    Used if the audio should be transcribed in a stream.
    """
    def __init__(self, audio_config=None):
        # inits all audio configs
        super(InferenceSpectrogramAudioParser, self).__init__(audio_config)

        self.n_fft = int(self.sampling_rate * self.window_size)
        self.hop_length = int(self.sampling_rate * self.window_stride)

        # These are estimated from the NST dataset.
        self.dataset_mean = 5.492418704733003
        self.dataset_std = 1.7552755216970917
        self.input_mean = 0
        self.input_std = 0
        self.alpha = 0
        self.alpha_increment = 0.1  # Corresponds to stop relying on dataset after 1 sec

        self.buffer = None
        self.has_buffer = False # Need this because of numpy arrays
        self.dummy_audio_buffer = []


    def parse_audio(self, part_of_recording, is_last=False):
        """
        Parses the given recording to a spectrogram for DanSpeech models.

        :param part_of_recording: Audio/Speech data in numpy array format.
        :param is_last: Indicating whether the part_of_recording is the last part of a recording
        :return: Adaoted Spectrogram
        """

        # If the last part is beneath the required size for stft, ignore it and reset
        # This is needed since we always want output for the last even if too short
        if is_last and len(part_of_recording) < self.n_fft:
            self.reset()
            return []

        self.dummy_audio_buffer.append(part_of_recording)

        # We need to save hop length for next iteration
        if self.has_buffer:
            part_of_recording = np.concatenate((self.buffer, part_of_recording), axis=None)

        # Left over samples for stft since we user "center" padding
        extra_samples = len(part_of_recording) % self.hop_length

        if extra_samples != 0:
            extra_samples_array = part_of_recording[-extra_samples:]
            part_of_recording = part_of_recording[:-extra_samples]

        self.buffer = part_of_recording[-self.hop_length:]

        # If we have extra samples, use them in buffer.
        if extra_samples != 0:
            self.buffer = np.concatenate((self.buffer, extra_samples_array), axis=None)

        self.has_buffer = True

        # Create the spectrogram
        D = librosa.stft(part_of_recording, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window=self.window, center=False)

        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)

        # Adaptive normalization parameters
        self.alpha += self.alpha_increment
        self.input_mean = (self.input_mean + np.mean(spect)) / 2
        self.input_std = (self.input_std + np.std(spect)) / 2

        # Whenever alpha is done, rely only on input stats
        if self.alpha < 1.0:
            mean = self.input_mean * self.alpha + (1 - self.alpha) * self.dataset_mean
            std = self.input_std * self.alpha + (1 - self.alpha) * self.dataset_std
        else:
            mean = self.input_mean
            std = self.input_std

        spect -= mean
        spect /= std
        spect = torch.FloatTensor(spect)

        return spect

    def reset(self):
        self.buffer = None
        self.has_buffer = False
        self.input_mean = 0
        self.input_std = 0
        self.alpha = 0
