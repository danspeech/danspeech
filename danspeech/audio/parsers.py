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

    def __init__(self, audio_config=None, data_augmenter=None):
        # inits all audio configs
        super(SpectrogramAudioParser, self).__init__(audio_config)

        self.n_fft = int(self.sampling_rate * self.window_size)
        self.hop_length = int(self.sampling_rate * self.window_stride)

        self.data_augmenter = data_augmenter

    def parse_audio(self, recording):

        if self.data_augmenter:
            recording = self.data_augmenter.augment(recording)

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

    def __init__(self, audio_config=None, context=20):
        # inits all audio configs
        super(InferenceSpectrogramAudioParser, self).__init__(audio_config)

        self.n_fft = int(self.sampling_rate * self.window_size)
        self.hop_length = int(self.sampling_rate * self.window_stride)
        self.context = context
        self.dataset_mean = 5.492418704733003
        self.dataset_std = 1.7552755216970917
        self.input_mean = 0
        self.input_std = 0
        self.alpha = 0
        self.alpha_increment = 0.1  # Corresponds to stop relying on dataset stats after 4sec
        self.nr_recordings = 0
        self.nr_frames = context * 2 + 5

    def parse_audio(self, part_of_recording, is_last=False):

        # Ignore last and
        if is_last and len(part_of_recording) < 320:
            if is_last:
                self.reset()
            return []

        self.alpha += self.alpha_increment

        D = librosa.stft(part_of_recording, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window=self.window, center=False)

        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)

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

        if is_last:
            self.reset()

        return spect

    def reset(self):
        self.input_mean = 0
        self.input_std = 0
        self.alpha = 0
        self.nr_recordings = 0
