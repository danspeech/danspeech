import audioop
import collections
import io
import json
import math
import os
import threading
import time

from danspeech.errors.recognizer_errors import ModelNotInitialized, WaitTimeoutError
from danspeech.DanSpeechRecognizer import DanSpeechRecognizer
from danspeech.audio.resources import SpeechSource, AudioData, SpeechFile
import numpy as np
import librosa


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

    def update_model(self, model):
        self.danspeech_recognizer.update_model(model)

    def update_decoder(self, lm=None, alpha=None, beta=None, beam_width=None):
        self.danspeech_recognizer.update_decoder(lm=lm, alpha=alpha, beta=beta, beam_width=beam_width)

    def listen(self, source, timeout=None, phrase_time_limit=None):
        """
        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.

        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.

        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, there will be no wait timeout.

        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.

        This operation will always complete within ``timeout + phrase_timeout`` seconds if both are numbers, either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = float(source.chunk) / source.sampling_rate
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer))  # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        # read audio input for phrases until there is a phrase that is long enough
        elapsed_time = 0  # number of seconds of audio read

        while True:
            frames = collections.deque()

            # store audio input until the phrase starts
            while True:
                # handle waiting too long for phrase by raising an exception
                elapsed_time += seconds_per_buffer
                if timeout and elapsed_time > timeout:
                    raise WaitTimeoutError("listening timed out while waiting for phrase to start")

                buffer = source.stream.read(source.chunk)
                if len(buffer) == 0: break  # reached end of the stream
                frames.append(buffer)
                if len(frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                    frames.popleft()

                # detect whether speaking has started on audio input
                energy = audioop.rms(buffer, source.sampling_width)  # energy of the audio signal
                if energy > self.energy_threshold: break

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
                if len(buffer) == 0: break  # reached end of the stream
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

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0: break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count): frames.pop()  # remove extra non-speaking frames at the end
        frame_data = b"".join(frames)

        return AudioData(frame_data, source.sampling_rate, source.sampling_width)


    def listen_stream(self, source, frames_first, frames_rest, timeout=None, phrase_time_limit=None):
        """
        Generator recording

        Records a single phrase from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance, which it returns.

        This is done by waiting until the audio has an energy above ``recognizer_instance.energy_threshold`` (the user has started speaking), and then recording until it encounters ``recognizer_instance.pause_threshold`` seconds of non-speaking or there is no more audio input. The ending silence is not included.

        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, there will be no wait timeout.

        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.

        This operation will always complete within ``timeout + phrase_timeout`` seconds if both are numbers, either by returning the audio data, or by raising a ``speech_recognition.WaitTimeoutError`` exception.
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

                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)

                if len(frames) > non_speaking_buffer_count:
                    # ensure we only keep the needed amount of non-speaking buffers
                    frames.pop(0)

                # detect whether speaking has started on audio input
                energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
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
                energy = audioop.rms(buffer, source.sampling_rate)  # unit energy of the audio signal within the buffer

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
        # obtain frame data
        frame_data = b"".join(frames)
        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH).get_array_data()

    def adjust_for_ambient_noise(self, source, duration=1):
        """
        Adjusts the energy threshold dynamically using audio from ``source`` (an ``AudioSource`` instance) to account for ambient noise.

        Intended to calibrate the energy threshold with the ambient energy level. Should be used on periods of audio without speech - will stop early if any speech is detected.

        The ``duration`` parameter is the maximum number of seconds that it will dynamically adjust the threshold for before returning. This value should be at least 0.5 in order to get a representative sample of the ambient noise.
        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before adjusting, see documentation for ``AudioSource``; are you using ``source`` outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = (source.chunk + 0.0) / source.sampling_rate
        elapsed_time = 0

        # adjust energy threshold until a phrase starts
        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration: break
            buffer = source.stream.read(source.chunk)
            energy = audioop.rms(buffer, source.sampling_width)  # energy of the audio signal

            # dynamically adjust the energy threshold using asymmetric weighted average
            damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer  # account for different chunk sizes and rates
            target_energy = energy * self.dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

    def listen_in_background(self, source, first_required_frames, general_required_frames):
        """
        Spawns a thread to repeatedly record phrases from ``source`` (an ``AudioSource`` instance) into an ``AudioData`` instance and call ``callback`` with that ``AudioData`` instance as soon as each phrase are detected.

        Returns a function object that, when called, requests that the background listener thread stop. The background thread is a daemon and will not stop the program from exiting if there are no other non-daemon threads. The function accepts one parameter, ``wait_for_stop``: if truthy, the function will wait for the background listener to stop before returning, otherwise it will return immediately and the background listener thread might still be running for a second or two afterwards. Additionally, if you are using a truthy value for ``wait_for_stop``, you must call the function from the same thread you originally called ``listen_in_background`` from.

        Phrase recognition uses the exact same mechanism as ``recognizer_instance.listen(source)``. The ``phrase_time_limit`` parameter works in the same way as the ``phrase_time_limit`` parameter for ``recognizer_instance.listen(source)``, as well.

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
                        for i in range(index+1):
                            data.pop(0)
                    should_try = False
                except IndexError:
                    time.sleep(0.2) # Wait 200ms and try again

            return is_last_, audio

        listener_thread = threading.Thread(target=threaded_listen)
        listener_thread.daemon = True
        listener_thread.start()
        return stopper, get_data

    def stop_microphone_streaming(self):
        print("Stopping microphone stream...")
        self.stream_thread_stopper(wait_for_stop=False)
        self.stream = False

    def microphone_streaming(self, source):
        self.danspeech_recognizer.enable_streaming()
        lookahead_context = self.danspeech_recognizer.model.context
        required_spec_frames = (lookahead_context - 1) * 2
        samples_pr_10ms = int(source.SAMPLE_RATE / 100)

        # First takes two samples pr 10ms, the rest needs 160 due to overlapping
        general_sample_requirement = samples_pr_10ms * 2 + (samples_pr_10ms * (required_spec_frames - 1))

        # First pass, we need more samples due to padding of initial conv layers
        first_samples_requirement = general_sample_requirement + (samples_pr_10ms * 15)

        samples_pr_frame = int(source.CHUNK)

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
                output = self.danspeech_recognizer.streaming_transcribe(data_array[:first_samples_requirement], is_last, is_first)
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
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using a loceal DanSpeech model.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the
        16 most likely beams from beam search with a language model

        If ``use_lm`` is false, the most likely transcription will be returned ignoring the ``show_all`` parameter

        """

        return self.danspeech_recognizer.transcribe(audio_data.get_array_data(),
                                                    show_all=show_all)
