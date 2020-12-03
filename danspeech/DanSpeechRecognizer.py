import warnings

import torch

from danspeech.deepspeech.decoder import GreedyDecoder, BeamCTCDecoder
from danspeech.errors.recognizer_errors import ModelNotInitialized
from danspeech.audio.parsers import SpectrogramAudioParser, InferenceSpectrogramAudioParser


class NoLmInstantiatedWarning(Warning):
    pass

class DanSpeechRecognizer(object):

    def __init__(self, model_name=None, lm_name=None,
                 alpha=1.3, beta=0.2, with_gpu=False,
                 beam_width=64):

        self.device = torch.device("cuda" if with_gpu else "cpu")
        print("Using device: {0}".format(self.device))

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
        self.audio_config = model.audio_conf
        self.model = model.to(self.device)
        self.model.eval()
        self.audio_parser = SpectrogramAudioParser(self.audio_config)

        self.labels = self.model.labels
        # When updating model, always update decoder because of labels
        self.update_decoder(labels=self.labels)

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


    def enable_streaming(self, secondary_model=None, return_string_parts=True):
        """
        Enables the DanSpeech system to perform speech recognition on a stream of audio data.

        :param secondary_model: A DanSpeech to perform speech recognition when a buffer of audio data has been build,
        hence this model can be given to provide better final transcriptions. If None, then the system will use the
        streaming model for the final output.
        """
        # Streaming declarations
        self.full_output = []
        self.iterating_transcript = ""
        if secondary_model:
            self.secondary_model = secondary_model.to(self.device)
            self.secondary_model.eval()
        else:
            self.secondary_model = None

        self.spectrograms = []

        # This is needed for streaming decoding
        self.greedy_decoder = GreedyDecoder(labels=self.labels, blank_index=self.labels.index('_'))

        # Use SpecroGramAudioParser
        self.audio_parser = InferenceSpectrogramAudioParser(audio_config=self.audio_config)

        if return_string_parts:
            self.string_parts = True
        else:
            self.string_parts = False

    def disable_streaming(self, keep_secondary_model=False):
        self.audio_parser = SpectrogramAudioParser(self.audio_config)
        self.greedy_decoder = None
        self.reset_streaming_params()
        self.string_parts = False

        if not keep_secondary_model:
            self.secondary_model = None


    def reset_streaming_params(self):
        self.iterating_transcript = ""
        self.full_output = []
        self.spectrograms = []


    def streaming_transcribe(self, recording, is_last, is_first):
        recording = self.audio_parser.parse_audio(recording, is_last)
        out = ""
        # This can happen if it is the last part of a recording and there is too little samples
        # to generate a spectrogram
        if len(recording) != 0:
            if self.secondary_model:
                self.spectrograms.append(recording)

            # Convert recording to batch for model purpose
            recording = recording.view(1, 1, recording.size(0), recording.size(1))
            recording = recording.to(self.device)

            out = self.model(recording, is_first, is_last)

            # First pass returns None, as we need more context to perform the first prediction
            if is_first:
                return ""

            self.full_output.append(out)

            # Decode the output with greedy decoding
            decoded_out, _ = self.greedy_decoder.decode(out)
            transcript = decoded_out[0][0]

            # Collapsing characters hack
            if self.iterating_transcript and transcript and self.iterating_transcript[-1] == transcript[0]:
                self.iterating_transcript = self.iterating_transcript + transcript[1:]
                transcript = transcript[1:]
            else:
                self.iterating_transcript += transcript

            if self.string_parts:
                out = transcript
            else:
                out = self.iterating_transcript

        if is_last:
            # If something was actually detected (require at least two characters)
            if len(self.iterating_transcript) > 1:

                # If we use secondary model, pass full output through the model
                if self.secondary_model:

                    final = torch.cat(self.spectrograms, dim=1)
                    self.spectrograms = []

                    final = final.view(1, 1, final.size(0), final.size(1))
                    final = final.to(self.device)
                    input_sizes = torch.IntTensor([final.size(3)]).int()
                    out, _ = self.secondary_model(final, input_sizes)
                    decoded_out, _ = self.decoder.decode(out)
                    decoded_out = decoded_out[0][0]

                    self.reset_streaming_params()
                    return decoded_out

                else:
                    # if no secondary model, check whether we need to decode with LM or not
                    if self.lm != "greedy":
                        final_out = torch.cat(self.full_output, dim=1)
                        decoded_out, _ = self.decoder.decode(final_out)
                        decoded_out = decoded_out[0][0]
                        self.reset_streaming_params()
                        return decoded_out
                    else:
                        out = self.iterating_transcript
                        self.reset_streaming_params()
                        return out
            else:
                return ""

        return out

    def transcribe(self, recording, show_all=False):
        recording = self.audio_parser.parse_audio(recording)
        recording = recording.view(1, 1, recording.size(0), recording.size(1))
        recording = recording.to(self.device)
        input_sizes = torch.IntTensor([recording.size(3)]).int()
        out, output_sizes = self.model(recording, input_sizes)
        decoded_output, _ = self.decoder.decode(out, output_sizes)
        if show_all:
            if self.lm == 'greedy':
                warnings.warn("You are trying to get all beams but no LM has been instantiated.",
                              NoLmInstantiatedWarning)
            return decoded_output[0]
        else:
            return decoded_output[0][0]
