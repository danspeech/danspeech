import audioop
import collections
import math
import threading
import time

from danspeech.errors.recognizer_errors import ModelNotInitialized, WaitTimeoutError
from danspeech.DanSpeechRecognizer import DanSpeechRecognizer
from danspeech.audio.resources import SpeechSource, AudioData
import numpy as np


class Recognizer(object):
    """
    Recognizer Class, which represents a collection of speech recognition functionality.
    """

    def __init__(self, model=None, lm=None, **kwargs):
        # minimum audio energy to consider for recording
        self.energy_threshold = 1000

        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5

        # seconds of non-speaking audio before a phrase is considered complete
        self.pause_threshold = 0.8

        # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
        self.phrase_threshold = 0.3
        # seconds of non-speaking audio to keep on both sides of the recording
        self.non_speaking_duration = 0.35

        # seconds after an internal operation (e.g., an API request) starts before it times out, or ``None`` for no timeout
        self.operation_timeout = None
        self.danspeech_recognizer = DanSpeechRecognizer(**kwargs)

        self.stream = False
        self.stream_thread_stopper = None

        if model:
            self.update_model(model)

        if lm:
            if not model:
                raise ModelNotInitialized("Trying to initialize language model without also choosing a DanSpeech "
                                          "acoustic model.")
            else:
                self.update_decoder(lm=lm)

        # Being able to bind the microphone to the recognizer is useful.
        self.microphone = None

    def update_model(self, model):
        """
        Updates the model being used by the Recognizer.

        :param model: DanSpeech model (see DanSpeech.pretrained_models)
        :return: None
        """
        self.danspeech_recognizer.update_model(model)
        print("DanSpeech model updated") #ToDO: Include model name

    def update_decoder(self, lm=None, alpha=None, beta=None, beam_width=None):
        """
        Updates the decoder being used by the Recognizer.

        If lm is None or "greedy", then the decoding will be performed by greedy decoding, and the alpha, beta and
        beam width parameters are therefore ignored.

        :param lm: DanSpeech Language model (see DanSpeech.language_models)
        :param alpha: Alpha parameter of beam search decoding. If None, then the decoder will use existing parameter
        in DanSpeechRecognizer.
        :param beta: Beta parameter of beam search decoding. If None, then the decoder will use existing parameter
        in DanSpeechRecognizer.
        :param beam_width: Beam width of beam search decoding. If None, then the decoder will use existing parameter
        in DanSpeechRecognizer.
        :return: None
        """
        self.danspeech_recognizer.update_decoder(lm=lm, alpha=alpha, beta=beta, beam_width=beam_width)
        print("DanSpeech decoder updated ") #ToDO: Include model name

    def listen_stream(self, source, frames_first, frames_rest, timeout=None, phrase_time_limit=None):
        """
        Adapted from: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

        Generator used to listen to the audio from a source e.g. a microphone. This generator is used
        by the streaming models.

        :param source: Source of audio. Needs to be a Danspeech.audio.resources.SpeechSource instance
        :param frames_first: Required frames before yielding data for the first pass to the streaming model
        :param frames_rest: Minimum required frames for passes after the first pass of the streaming model.
        :param timeout: Maximum number of seconds that this will wait until a phrase starts
        :param phrase_time_limit: Maxumum number of seconds to that will allow a phrase to continue before stopping
        :return: Data and an indicator whether it is the last part of a streaming part
        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = float(source.chunk) / source.sampling_rate
        pause_buffer_count = int(math.ceil(
            self.pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count = int(math.ceil(
            self.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(
            self.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0  # number of seconds of audio read
        is_first = True
        while True:
            frames = []

            # store audio input until the phrase starts
            while True:
                # handle waiting too long for phrase by raising an exception
                elapsed_time += seconds_per_buffer
                if timeout and elapsed_time > timeout:
                    raise WaitTimeoutError("listening timed out while waiting for phrase to start")

                buffer = source.stream.read(source.chunk)
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)

                if len(frames) > non_speaking_buffer_count:
                    # ensure we only keep the needed amount of non-speaking buffers
                    frames.pop(0)

                # detect whether speaking has started on audio input
                energy = audioop.rms(buffer, source.sampling_width)  # energy of the audio signal
                if energy > self.energy_threshold:
                    break

                # dynamically adjust the energy threshold using asymmetric weighted average
                if self.dynamic_energy_threshold:
                    damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
                    target_energy = energy * self.dynamic_energy_ratio
                    self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time
            while True:
                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

                buffer = source.stream.read(source.chunk)
                if len(buffer) == 0:
                    break  # reached end of the stream

                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                energy = audioop.rms(buffer, source.sampling_width)  # unit energy of the audio signal within the buffer

                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:  # end of the phrase
                    break

                if is_first:
                    if len(frames) == frames_first:
                        is_first = False
                        yield False, self.get_audio_data(frames, source)
                        frames = []
                else:
                    if len(frames) == frames_rest:
                        yield False, self.get_audio_data(frames, source)
                        frames = []

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0:
                break  # phrase is long enough or we've reached the end of the stream, so stop listening

        if not frames:
            yield True, []

        frame_data = b"".join(frames)

        yield True, AudioData(frame_data, source.sampling_rate, source.sampling_width).get_array_data()

    @staticmethod
    def get_audio_data(frames, source):
        """
        Function to convert the frames (bytes) from a stream to an array used for DanSpeech models

        :param frames: Byte frames
        :param source: Source of stream/frames
        :return: Numpy array with speech data
        """
        # obtain frame data
        frame_data = b"".join(frames)
        return AudioData(frame_data, source.sampling_rate, source.sampling_width).get_array_data()

    def adjust_for_ambient_noise(self, source, duration=1):
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
        Modified for DanSpeech

        :param source: Source of audio. Needs to be a Danspeech.audio.resources.SpeechSource instance
        :param duration: Maximum duration of adjusting the energy threshold
        :return: None
        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before adjusting, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = (source.chunk + 0.0) / source.sampling_rate
        elapsed_time = 0

        # adjust energy threshold until a phrase starts
        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration:
                break

            buffer = source.stream.read(source.chunk)
            energy = audioop.rms(buffer, source.sampling_width)  # energy of the audio signal

            # dynamically adjust the energy threshold using asymmetric weighted average
            damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
            target_energy = energy * self.dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

    def listen_in_background(self, source, first_required_frames, general_required_frames):
        """
        Spawns a thread which listens to the source of data

        :param source: Source of stream/frames
        :param first_required_frames: Required frames before yielding data for the first pass to the streaming model
        :param general_required_frames: Minimum required frames for passes after the first pass of the streaming model.

        :return: Stopper function used to stop the thread, and a data_getter which returns data from the thread
        according current steps.
        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"

        # These act as globals variables for thread helper functions
        running = [True]
        data = []
        processed_background = [0]
        processed_data_getter = [0]

        def threaded_listen():
            # Trhead to run in background
            processed = 0
            with source as s:
                while running[0]:
                    generator = self.listen_stream(s, first_required_frames, general_required_frames)
                    try:  # Listen until silence has been detected
                        while True:
                            is_last_, temp = next(generator)
                            data.append((is_last_, temp))

                            # If is last, we start new listen generator
                            if is_last_:
                                processed += 1
                                processed_background[0] = processed
                                break

                    except WaitTimeoutError:  # listening timed out, just try again
                        pass

        def stopper(wait_for_stop=True):
            running[0] = False

            if wait_for_stop:
                listener_thread.join()  # block until the background thread is done, which can take around 1 second

        def get_data(index):
            # We need while loop in case data processing (model prediction) is faster than listening
            should_try = True

            # If background thread is ahead, get all data and predict
            if processed_background[0] > processed_data_getter[0]:
                first = True
                while should_try:
                    if first:
                        is_last_, audio = data[index]
                        first = False
                    else:
                        is_last_, temp = data[index]
                        audio = np.concatenate((audio, temp))

                    if not is_last_:
                        index += 1
                    else:
                        should_try = False

                # Clean up
                for i in range(index + 1):
                    data.pop(0)

                processed_data_getter[0] += 1
                return is_last_, audio

            while should_try:
                try:
                    is_last_, audio = data[index]
                    # If ending, clean up data that has already been processed
                    if is_last_:
                        processed_data_getter[0] += 1
                        for i in range(index + 1):
                            data.pop(0)
                    should_try = False
                except IndexError:
                    time.sleep(0.2)  # Wait 200ms and try again

            return is_last_, audio

        listener_thread = threading.Thread(target=threaded_listen)
        listener_thread.daemon = True
        listener_thread.start()
        return stopper, get_data

    def stop_microphone_streaming(self):
        """
        Used to stop microphone streaming.

        :return: None
        """
        if self.stream:
            print("Stopping microphone stream...")
            self.stream_thread_stopper(wait_for_stop=False)
            self.stream = False
        else:
            print("No stream is running for the Recognizer")

    def microphone_streaming(self, source):
        """
        Generator class to stream from the source of audio.

        This class handles the correct amounts needed by the streamer model. If the current held by DanSpeechRecognizer
        is not a streaming model, the function will init the streamingCPU model.

        :param source: Source of audio
        :return: Boolean to indicated if it is ending of an utterance and the transcribed output
        """
        self.danspeech_recognizer.enable_streaming()
        lookahead_context = self.danspeech_recognizer.model.context
        required_spec_frames = (lookahead_context - 1) * 2
        samples_pr_10ms = int(source.sampling_rate / 100)

        # First takes two samples pr 10ms, the rest needs 160 due to overlapping
        general_sample_requirement = samples_pr_10ms * 2 + (samples_pr_10ms * (required_spec_frames - 1))

        # First pass, we need more samples due to padding of initial conv layers
        first_samples_requirement = general_sample_requirement + (samples_pr_10ms * 15)

        samples_pr_frame = int(source.chunk)

        # Init general required frames from source
        counter = 0
        while counter * samples_pr_frame < general_sample_requirement:
            counter += 1

        general_required_frames = counter

        while counter * samples_pr_frame < first_samples_requirement:
            counter += 1

        first_required_frames = counter

        data_array = None
        data_counter = 0
        iterator = 0
        is_first = True
        self.stream = True

        stopper, data_getter = self.listen_in_background(source, first_required_frames, general_required_frames)
        self.stream_thread_stopper = stopper

        while self.stream:
            if is_first:
                is_last, data_array = data_getter(data_counter)
                data_counter += 1
            else:
                is_last, temp = data_getter(data_counter)
                data_counter += 1
                data_array = np.concatenate((data_array, temp))
                nr_data_points = len(data_array) - iterator
                data_rest = nr_data_points % samples_pr_10ms
                consume = nr_data_points - data_rest

            if is_first:
                output = self.danspeech_recognizer.streaming_transcribe(data_array[:first_samples_requirement], is_last,
                                                                        is_first)
                iterator += first_samples_requirement - samples_pr_10ms
                is_first = False
            elif is_last:
                output = self.danspeech_recognizer.streaming_transcribe(data_array[iterator:], is_last, is_first)
            else:
                output = self.danspeech_recognizer.streaming_transcribe(data_array[iterator:iterator + consume],
                                                                        is_last, is_first)
                iterator += consume - samples_pr_10ms

            if is_last and not output:
                yield is_last, None
            elif output:
                yield is_last, output

            # Reset parameters
            if is_last:
                data_counter = 0
                iterator = 0
                is_first = True
                data_array = None

    def recognize(self, audio_data, show_all=False):
        """
        Performs speech recognition with the current initialized model.

        :param audio_data: Numpy array of audio data
        :param show_all: Whether to return all beams for beam search, if the beam search is enabled.
        :return: Returns the most likely transcription if show_all is false (the default). Otherwise, returns the
        most likely beams from beam search with a language model
        """

        return self.danspeech_recognizer.transcribe(audio_data, show_all=show_all)
