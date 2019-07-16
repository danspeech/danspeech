import torch

from danspeech.deepspeech.decoder import GreedyDecoder, BeamCTCDecoder
from danspeech.errors.recognizer_errors import ModelNotInitialized
from danspeech.audio.parsers import SpectrogramAudioParser


class DanSpeechRecognizer(object):

    def __init__(self, model_name=None, lm_name=None,
                 alpha=1.3, beta=0.2, with_gpu=False,
                 beam_width=64):

        self.device = torch.device("cuda" if with_gpu else "cpu")

        # Init model if given
        if model_name:
            self.update_model(model_name)
        else:
            self.model = None
            self.model_name = None
            self.labels = None
            self.audio_config = None
            self.audio_parser = None

        # Always set alpha and beta
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width

        # Init LM if given
        if lm_name:
            if not self.model:
                raise ModelNotInitialized("Trying to initialize LM without also choosing a DanSpeech model.")
            else:
                self.update_decoder(lm_name)
                self.lm = lm_name
        else:
            self.lm = None
            self.decoder = None

    def update_model(self, model):
        self.model = model.to(self.device)
        self.model.eval()
        self.audio_config = self.model.audio_conf
        self.audio_parser = SpectrogramAudioParser(self.audio_config)

        # When updating model, always update decoder because of labels
        self.update_decoder(labels=self.model.labels)

    def update_decoder(self, lm=None, alpha=None, beta=None, labels=None, beam_width=None):

        update = False

        # If both lm_name and decoder is not set, then we need to init greedy as default use
        if not self.lm and not self.decoder:
            update = True
            self.lm = "greedy"

        if lm and self.lm != lm:
            update = True
            self.lm = lm

        if alpha and self.alpha != alpha:
            update = True
            self.alpha = alpha

        if beta and self.beta != beta:
            update = True
            self.beta = beta

        if labels and labels != self.labels:
            update = True
            self.labels = labels

        if beam_width and beam_width != self.beam_width:
            update = True
            self.beam_width = beam_width

        if update:
            if self.lm != "greedy":
                self.decoder = BeamCTCDecoder(labels=self.labels, lm_path=self.lm,
                                              alpha=self.alpha, beta=self.beta,
                                              beam_width=self.beam_width, num_processes=6, cutoff_prob=1.0,
                                              cutoff_top_n=40, blank_index=self.labels.index('_'))

            else:
                self.decoder = GreedyDecoder(labels=self.labels, blank_index=self.labels.index('_'))

    def transcribe(self, recording, show_all=False):
        recording = self.audio_parser.parse_audio(recording)
        recording = recording.view(1, 1, recording.size(0), recording.size(1))
        recording.to(self.device)
        input_sizes = torch.IntTensor([recording.size(3)]).int()
        out, output_sizes = self.model(recording, input_sizes)
        decoded_output, _ = self.decoder.decode(out, output_sizes)

        if show_all:
            return decoded_output[0]
        else:
            return decoded_output[0][0]
