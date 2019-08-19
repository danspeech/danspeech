import torch

from danspeech.deepspeech.decoder import GreedyDecoder, BeamCTCDecoder
from danspeech.errors.recognizer_errors import ModelNotInitialized
from danspeech.audio.parsers import SpectrogramAudioParser, InferenceSpectrogramAudioParser
from danspeech.pretrained_models import DanSpeechPrimary
from danspeech.language_models import DSL3gram

class DanSpeechRecognizer(object):

    def __init__(self, model_name=None, lm_name=None,
                 alpha=1.3, beta=0.2, with_gpu=False,
                 beam_width=64):

        self.device = torch.device("cuda" if with_gpu else "cpu")
        print(self.device)

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

        # Streaming declarations
        self.streaming = False
        self.full_output = []
        self.iterating_transcript = ""

        self.second_model = DanSpeechPrimary()
        self.audio_config = self.second_model.audio_conf
        self.second_model =  self.second_model.to(self.device)
        self.second_model.eval()

        self.second_decoder = DSL3gram()

        # When updating model, always update decoder because of labels
        self.second_decoder = decoder = BeamCTCDecoder(labels=self.second_model.labels, lm_path=self.second_decoder,
                                      alpha=self.alpha, beta=self.beta,
                                      beam_width=self.beam_width, num_processes=6, cutoff_prob=1.0,
                                      cutoff_top_n=40, blank_index=self.labels.index('_'))

    def update_model(self, model):
        self.audio_config = model.audio_conf
        self.model = model.to(self.device)
        self.model.eval()
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

    def enable_streaming(self):
        self.streaming = True
        self.greedy_decoder = GreedyDecoder(labels=self.labels, blank_index=self.labels.index('_'))
        self.audio_parser = InferenceSpectrogramAudioParser(audio_config=self.audio_config)

    def disable_streaming(self):
        self.streaming = False
        self.audio_parser = SpectrogramAudioParser(self.audio_config)
        self.iterating_transcript = ""
        self.full_output = []
        self.spectrograms = []

    def streaming_transcribe(self, recording, is_last, is_first):
        recording = self.audio_parser.parse_audio(recording, is_last)

        if len(recording) != 0:
            # ToDO: Remove but keep here for now
            self.spectrograms.append(recording)
            # Convert recording to batch for model purpose
            recording = recording.view(1, 1, recording.size(0), recording.size(1))

            recording.to(self.device)
            out = self.model(recording, is_first, is_last)

            # First pass returns None, as we need more context for first prediction
            if is_first:
                return ""

            self.full_output.append(out)
            decoded_out, _ = self.greedy_decoder.decode(out)
            transcript = decoded_out[0][0]

            # Collapsing characters hack
            if self.iterating_transcript and transcript and self.iterating_transcript[-1] == transcript[0]:
                self.iterating_transcript = self.iterating_transcript[:-1] + transcript
            else:
                self.iterating_transcript += transcript

        if is_last:
            # ToDO: Remove but keep here for now
            final = torch.cat(self.spectrograms, dim=1)
            #plt.imshow(final)
            #plt.colorbar()
            #plt.show()
            #self.spectrograms = []
            if self.lm != "greedy":
                final_out = torch.cat(self.full_output, dim=1)
                decoded_out, _ = self.decoder.decode(final_out)
                decoded_out = decoded_out[0][0]
                output = ""
                if len(decoded_out) > 1:
                    output = str(decoded_out[0]).upper() + decoded_out[1:] + ".\n"
                self.full_output = []
                self.iterating_transcript = ""
                return output
            else:
                output = ""
                if len(self.iterating_transcript) > 1:
                    out, _ = self.second_model(final)
                    decoded_out, _ = self.decoder.decode(out)
                    decoded_out = decoded_out[0][0]
                    output = decoded_out
                    #output = str(self.iterating_transcript[0]).upper() + self.iterating_transcript[1:] + ".\n"
                self.iterating_transcript = ""
                return output

        return self.iterating_transcript

    def transcribe(self, recording, show_all=False):
        recording = self.audio_parser.parse_audio(recording)
        recording = recording.view(1, 1, recording.size(0), recording.size(1))
        recording = recording.to(self.device)
        input_sizes = torch.IntTensor([recording.size(3)]).int()
        out, output_sizes = self.model(recording, input_sizes)
        decoded_output, _ = self.decoder.decode(out, output_sizes)

        if show_all:
            return decoded_output[0]
        else:
            return decoded_output[0][0]
