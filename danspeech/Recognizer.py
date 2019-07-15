import io
import json

from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from danspeech.errors.recognizer_errors import UnknownValueError, RequestError, ModelNotInitialized
from danspeech.DanSpeechRecognizer import DanSpeechRecognizer
from danspeech.audio.resources import SpeechSource, AudioData, SpeechFile


class Recognizer(object):

    def __init__(self, model=None, lm=None, **kwargs):
        """
        Creates a new ``Recognizer`` instance, whi  ch represents a collection of speech recognition functionality.

        lm_name requires model_name.
        alpha and beta requires lm

        """
        # minimum audio energy to consider for recording
        self.energy_threshold = 300

        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5

        # seconds of non-speaking audio before a phrase is considered complete
        self.pause_threshold = 0.8

        # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        self.phrase_threshold = 0.3
        # seconds of non-speaking audio to keep on both sides of the recording
        self.non_speaking_duration = 0.5

        # seconds after an internal operation (e.g., an API request) starts before it times out, or ``None`` for no timeout
        self.operation_timeout = None
        self.danspeech_recognizer = DanSpeechRecognizer(**kwargs)

        if model:
            self.update_model(model)

        if lm:
            if not model:
                    raise ModelNotInitialized("Trying to initialize language model without also choosing a DanSpeech "
                                              "acoustic model.")
            else:
                self.update_decoder(lm=lm)

    def update_model(self, model):
        self.danspeech_recognizer.update_model(model)

    def update_decoder(self, lm=None, alpha=None, beta=None, beam_width=None):
        self.danspeech_recognizer.update_decoder(lm=lm, alpha=alpha, beta=beta, beam_width=beam_width)

    def record(self, source, duration=None, offset=None):
        """
        Records up to ``duration`` seconds of audio from ``source`` (an ``AudioSource`` instance) starting at ``offset`` (or at the beginning if not specified) into an ``AudioData`` instance, which it returns.

        If ``duration`` is not specified, then it will record until there is no more audio input.
        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before recording, see documentation for " \
                                          "``AudioSource``; are you using ``source`` outside of a ``with`` statement? "

        frames_bytes = io.BytesIO()
        seconds_per_buffer = (source.CHUNK + 0.0) / source.SAMPLE_RATE
        elapsed_time = 0
        offset_time = 0
        offset_reached = False
        while True:  # loop for the total number of chunks needed
            if offset and not offset_reached:
                offset_time += seconds_per_buffer
                if offset_time > offset:
                    offset_reached = True

            buffer = source.stream.read(source.CHUNK)
            if len(buffer) == 0:
                break

            if offset_reached or not offset:
                elapsed_time += seconds_per_buffer
                if duration and elapsed_time > duration:
                    break

                frames_bytes.write(buffer)

        frame_data = frames_bytes.getvalue()
        frames_bytes.close()
        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

    def recognize_google(self, audio_data, key=None, language="da-DK", pfilter=0, show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Google Speech Recognition API.

        The Google Speech Recognition API key is specified by ``key``. If not specified, it uses a generic key that works out of the box. This should generally be used for personal or testing purposes only, as it **may be revoked by Google at any time**.

        To obtain your own API key, simply following the steps on the `API Keys <http://www.chromium.org/developers/how-tos/api-keys>`__ page at the Chromium Developers site. In the Google Developers Console, Google Speech Recognition is listed as "Speech API".

        The recognition language is determined by ``language``, an RFC5646 language tag like ``"en-US"`` (US English) or ``"fr-FR"`` (International French), defaulting to US English. A list of supported language tags can be found in this `StackOverflow answer <http://stackoverflow.com/a/14302134>`__.

        The profanity filter level can be adjusted with ``pfilter``: 0 - No filter, 1 - Only shows the first character and replaces the rest with asterisks. The default is level 0.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the raw API response as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
        assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
        assert key is None or isinstance(key, str), "``key`` must be ``None`` or a string"
        assert isinstance(language, str), "``language`` must be a string"

        flac_data = audio_data.get_flac_data(
            convert_rate=None if audio_data.sample_rate >= 8000 else 8000,  # audio samples must be at least 8 kHz
            convert_width=2  # audio samples must be 16-bit
        )
        if key is None: key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
        url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
            "client": "chromium",
            "lang": language,
            "key": key,
            "pFilter": pfilter
        }))
        request = Request(url, data=flac_data,
                          headers={"Content-Type": "audio/x-flac; rate={}".format(audio_data.sample_rate)})

        # obtain audio transcription results
        try:
            response = urlopen(request, timeout=self.operation_timeout)
        except HTTPError as e:
            raise RequestError("recognition request failed: {}".format(e.reason))
        except URLError as e:
            raise RequestError("recognition connection failed: {}".format(e.reason))
        response_text = response.read().decode("utf-8")

        # ignore any blank blocks
        actual_result = []
        for line in response_text.split("\n"):
            if not line: continue
            result = json.loads(line)["result"]
            if len(result) != 0:
                actual_result = result[0]
                break

        # return results
        if show_all: return actual_result
        if not isinstance(actual_result, dict) or len(
                actual_result.get("alternative", [])) == 0: raise UnknownValueError()

        if "confidence" in actual_result["alternative"]:
            # return alternative with highest confidence score
            best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
        else:
            # when there is no confidence available, we arbitrarily choose the first hypothesis.
            best_hypothesis = actual_result["alternative"][0]
        if "transcript" not in best_hypothesis:
            raise UnknownValueError()
        return best_hypothesis["transcript"]

    def recognize(self, audio_data, show_all=False):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using a loceal DanSpeech model.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the
        16 most likely beams from beam search with a language model

        If ``use_lm`` is false, the most likely transcription will be returned ignoring the ``show_all`` parameter

        """

        return self.danspeech_recognizer.transcribe(audio_data.get_array_data(),
                                                    show_all=show_all)
