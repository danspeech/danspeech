import torch

import scipy
import numpy as np
import librosa

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class AudioParser(object):
    """
    Abstract class for AudioParsers
    """

    def __init__(self, audio_config=None):
        self.audio_config = audio_config
        if not self.audio_config:
            self.audio_config = {}

        # Defaulting if no audio_config
        self.normalize = self.audio_config.get("normalize", True)
        self.sample_rate = self.audio_config.get("sample_rate", 16000)
        window = self.audio_config.get("window", "hamming")
        self.window = windows[window]

        self.window_stride = self.audio_config.get("window_stride", 0.01)
        self.window_size = self.audio_config.get("window_size", 0.02)

    def parse_audio(self, recording):
        raise NotImplementedError


class SpectrogramAudioParser(AudioParser):

    def __init__(self, audio_config=None):
        # inits all audio configs
        super(SpectrogramAudioParser, self).__init__(audio_config)

        self.n_fft = int(self.sample_rate * self.window_size)
        self.hop_length = int(self.sample_rate * self.window_stride)

    def parse_audio(self, recording):
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
