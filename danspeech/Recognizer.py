import audioop
import collections
import math
import threading
import time

from danspeech.errors.recognizer_errors import ModelNotInitialized, WaitTimeoutError, WrongUsageOfListen, NoDataInBuffer
from danspeech.DanSpeechRecognizer import DanSpeechRecognizer
from danspeech.audio.resources import SpeechSource, AudioData
import numpy as np


class Recognizer(object):
    """
    Recognizer Class, which represents a collection of speech recognition functionality.

    None of the parameters are required, but you need to update the Recognizer with a valid model
    before being able to perform speech recognition.

        :param DeepSpeech model:
            A valid DanSpeech model (``danspeech.deepspeech.model.DeepSpeech``)

            See :ref:`pre-trained-models` for more information.

            This can also be your custom DanSpeech trained model.
        :param str lm:
            A path (``str``) to a valid .klm language model. See :ref:`language-models` for a list
            of pretrained available models.
        :param bool with_gpu:
            A ``bool`` representing whether you want to run the ``Recognizer`` with a GPU.

            Note: Requires a GPU.
        :param \**kwargs:
            Additional decoder arguments. See :meth:`Recognizer.update_decoder` for more information.
        :Example:
            recognizer = Recognizer()
    """

    def __init__(self, model=None, lm=None, with_gpu=False, **kwargs):


        # Listening to a stream parameters
        # minimum audio energy to consider for recording
        self.energy_threshold = 1000

        # seconds of non-speaking audio before a phrase is considered complete
        self.pause_threshold = 0.8

        # minimum seconds of speaking audio before we consider the speaking audio a phrase
        # values below this are ignored (for filtering out clicks and pops)
        self.phrase_threshold = 0.3

        # seconds of non-speaking audio to keep on both sides of the recording
        self.non_speaking_duration = 0.35

        # Seconds before we consider a clip and actual clip
        self.mininum_required_speaking_seconds = 0.7

        # Adjust energy params
        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5

        self.danspeech_recognizer = DanSpeechRecognizer(with_gpu=with_gpu, **kwargs)

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

    def recognize(self, audio_data, show_all=False):
        """
        Performs speech recognition with the current initialized DanSpeech model
        (``danspeech.deepspeech.model.DeepSpeech``).

        :param array audio_data: ``Numpy array`` of audio data. Use :meth:`audio.load_audio` to load your audio
            into a valid format.
        :param bool show_all: Whether to return all beams from beam search, if decoding is
            performed with a language model.
        :return: The most likely transcription if ``show_all=false`` (the default). Otherwise, returns the
            most likely beams from beam search with a language model.
        :rtype: str or list[str] if ``show_all=True``.
        """
        return self.danspeech_recognizer.transcribe(audio_data, show_all=show_all)

    def update_model(self, model):
        """
        Updates the DanSpeech model being used by the Recognizer.

        :param model: A valid DanSpeech model (``danspeech.deepspeech.model.DeepSpeech``).
            See :ref:`pre-trained-models` for a list of pretrained available models.
            This can also be your custom DanSpeech trained model.

        """
        self.danspeech_recognizer.update_model(model)
        print("DanSpeech model updated to: {0}".format(model.model_name))

    def update_decoder(self, lm=None, alpha=None, beta=None, beam_width=None):
        """
        Updates the decoder being used by the Recognizer. By default, greedy decoding of the DanSpeech
        model will be performed.

        If lm is None or lm="greedy", then the decoding will be performed by greedy decoding, and the alpha, beta and
        beam width parameters are therefore ignored.

        **Warning:** Language models requires the `ctc-decode <https://github.com/parlance/ctcdecode.git>`_
        python package to work.

        :param str lm: A path to a valid .klm language model. See :ref:`language-models` for a list
            of pretrained available models.
        :param float alpha: Alpha parameter of beam search decoding.  If None, then the default parameter of
            ``alpha=1.3`` is used
        :param float beta: Beta parameter of beam search decoding. If None, then the default parameter of
            ``beta=0.2`` is used
        :param int beam_width: Beam width of beam search decoding. If None, then the default parameter of
            ``beam_width=64`` is used.
        """
        self.danspeech_recognizer.update_decoder(lm=lm, alpha=alpha, beta=beta, beam_width=beam_width)
        print("DanSpeech decoder updated ")  # ToDO: Include model name


    def listen(self, source, timeout=None, phrase_time_limit=None):
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
        Modified for DanSpeech.

        Listens to a stream of audio.

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
                if len(
                        frames) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
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
            if phrase_count >= phrase_buffer_count or len(
                    buffer) == 0: break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for i in range(
                pause_count - non_speaking_buffer_count): frames.pop()  # remove extra non-speaking frames at the end
        frame_data = b"".join(frames)

        return AudioData(frame_data, source.sampling_rate, source.sampling_width)

    def listen_stream(self, source, timeout=None, phrase_time_limit=None):
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
        # ToDO: Change the assertions
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before listening, " \
                                          "see documentation for ``AudioSource``; are you using " \
                                          "``source`` outside of a ``with`` statement?"
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
        buffer = []
        while self.stream:
            frames = []

            # store audio input until the phrase starts
            while True and self.stream:
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

            # If streaming has stopped while looking for speech, break out of thread so it can stop
            if not self.stream:
                yield False, []

            # Yield the silence in the beginning
            yield False, frames

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            phrase_start_time = elapsed_time
            while True:

                buffer = source.stream.read(source.chunk)
                if len(buffer) == 0:
                    break  # reached end of the stream

                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer
                if phrase_time_limit and elapsed_time - phrase_start_time > phrase_time_limit:
                    break

                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                energy = audioop.rms(buffer, source.sampling_width)  # unit energy of the audio signal within the buffer

                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1

                if pause_count > pause_buffer_count:  # end of the phrase
                    break

                # If data is being processed
                yield False, buffer

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0:
                break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # Ending of stream, should start a new stream
        if len(buffer) == 0:
            yield True, []
        else:
            yield True, buffer

        # If we go here, then it is wrong usage of stream
        raise WrongUsageOfListen("Wrong usage of stream. Overwrite the listen generator with a new generator instance"
                                 "since this instance has completed a full listen.")

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

    def listen_in_background(self, source):
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

        def threaded_listen():
            # Thread to run in background
            with source as s:
                while running[0]:
                    generator = self.listen_stream(s)
                    try:  # Listen until stream detects silence
                        while True:
                            is_last_, temp = next(generator)
                            if isinstance(temp, list):
                                temp = self.get_audio_data(temp, source)
                            else:
                                temp = self.get_audio_data([temp], source)

                            # Append data
                            data.append((is_last_, temp))

                            # If is last, we start new listen generator
                            if is_last_:
                                break

                    except WaitTimeoutError:  # listening timed out, just try again
                        pass

        def stopper(wait_for_stop=True):
            running[0] = False

            if wait_for_stop:
                listener_thread.join()  # block until the background thread is done, which can take around 1 second

        def get_data():
            while True:
                try:
                    is_last_, audio = data[0]
                    # Remove from buffer
                    data.pop(0)
                    break
                except IndexError:
                    raise NoDataInBuffer

            return is_last_, audio

        listener_thread = threading.Thread(target=threaded_listen)
        listener_thread.daemon = True
        listener_thread.start()
        return stopper, get_data

    def enable_streaming(self):
        """
        Adjusts the Recognizer to continuously transcribe a stream of audio input.

        Use this before starting a stream.

        :example:

            .. code-block:: python

                recognizer.enable_streaming()

        """
        if self.stream:
            print("Streaming already enabled...")
        else:
            self.stream = True

    def disable_streaming(self):
        """
        Adjusts the Recognizer to stop expecting a stream of audio input.

        Use this after cancelling a stream.

        :example:

            .. code-block:: python

                recognizer.disable_streaming()

        """
        if self.stream:
            self.stream = False
            self.stream_thread_stopper(wait_for_stop=False)
        else:
            self.stream = True

    def streaming(self, source):
        """
        Generator class for a stream audio source a :meth:`Microphone`

        Spawns a background thread and uses the loaded model to continuously transcribe audio input between
        detected silences from the :meth:`Microphone` stream.

        **Warning:** Requires that :meth:`Recognizer.enable_streaming` has been called.

        :param Microphone source: Source of audio.

        :example:

            .. code-block:: python

                generator = recognizer.streaming(source=m)

                # Runs for a long time. Insert your own stop condition.
                for i in range(100000):
                    trans = next(generator)
                    print(trans)

        """
        stopper, data_getter = self.listen_in_background(source)
        self.stream_thread_stopper = stopper

        is_last = False
        is_first_data = False
        data_array = []

        while self.stream:
            # Loop for data (gets all the available data from the stream)
            while True:

                # If it is the last one in a stream, break and perform recognition no matter what
                if is_last:
                    is_first_data = True
                    break

                # Get all available data
                try:
                    if is_first_data:
                        is_last, data_array = data_getter()
                        is_first_data = False
                    else:
                        is_last, temp = data_getter()
                        data_array = np.concatenate((data_array, temp))
                # If this exception is thrown, then we no available data
                except NoDataInBuffer:
                    # If no data in buffer, we sleep and wait
                    time.sleep(0.2)

            # Since we only break out of data loop, if we need a prediction, the following works
            # We only do a prediction if the length of gathered audio is above a threshold
            if len(data_array) > self.mininum_required_speaking_seconds * source.sampling_rate:
                yield self.recognize(data_array)

            is_last = False
            data_array = []

    def enable_real_time_streaming(self, streaming_model, secondary_model=None, string_parts=True):
        """
        Adjusts the Recognizer to  continuously transcribe a stream of audio input real-time.

        Real-time audio streaming utilize a uni-directional model to transcribe an utterance while
        being spoken in contrast to :meth:`Recognizer.streaming`, where the utterance is transcribed after
        a silence has been detenced.

        Use this before starting a (:meth:`Recognizer.real_time_streaming`) stream.

        :param DeepSpeech streaming_model: The DanSpeech model to perform streaming. This model needs to
            be uni-directional. This is required for real-time streaming to work.
            The two available DanSpeech models are :meth:`pretrained_models.CPUStreamingRNN` and
            :meth:`pretrained_models.GPUStreamingRNN` but you may use your own custom streaming model as well.

        :param DeepSpeech secondary_model: A valid DanSpeech model (``danspeech.deepspeech.model.DeepSpeech``).
            The secondary model transcribes the output after a silence is detected. This is useful since the performance
            of uni-directional models is very poor compared to bi-directional models, which require the full utterance.
            See :ref:`pre-trained-models` for more information on available models. This can also be your custom DanSpeech trained model.

        :param bool string_parts:
            Boolean indicating whether you want the generator (:meth:`Recognizer.real_time_streaming`) to yield
            parts of the string or the whole (currrent) iterating string. Recommended is ``string_parts=True`` and then
            keep track of the transcription yourself.
        :example:

            .. code-block:: python

                recognizer.enable_real_time_streaming(streaming_model=CPUStreamingRNN())

        """
        # Update streaming model from Recognizer and not inside the DanSpeechRecognizer
        self.update_model(streaming_model)
        self.danspeech_recognizer.enable_streaming(secondary_model, string_parts)
        self.stream = True

    def disable_real_time_streaming(self, keep_secondary_model_loaded=False):
        """
        Adjusts the Recognizer to stop expecting a stream of audio input.

        Use this after cancelling a stream.

        :param bool keep_secondary_model_loaded: Whether to keep the secondary model in memory or not. Generally,
            you do not want to keep it in memory, unless you want to perform :meth:`Recognizer.real_time_streaming`
            again after disabling the real-time streaming.

        :example:

            .. code-block:: python

                recognizer.disable_real_time_streaming()

        """
        if self.stream:
            print("Stopping microphone stream...")
            self.stream = False
            self.stream_thread_stopper(wait_for_stop=False)
            self.danspeech_recognizer.disable_streaming(keep_secondary_model=keep_secondary_model_loaded)
        else:
            print("No stream is running for the Recognizer")

    def real_time_streaming(self, source):
        """
        Generator class for a real-time stream audio source a :meth:`Microphone`.

        Spawns a background thread and uses the loaded model(s) to continuously transcribe an audio utterance
        while it is being spoken.

        **Warning:** Requires that :meth:`Recognizer.enable_real_time_streaming` has been called.

        :param Microphone source: Source of audio.

        :example:

            .. code-block:: python

                generator = r.real_time_streaming(source=m)

                iterating_transcript = ""
                print("Speak!")
                while True:
                    is_last, trans = next(generator)

                    # If the transcription is empty, it means that the energy level required for data
                    # was passed, but nothing was predicted.
                    if is_last and trans:
                        print("Final: " + trans)
                        iterating_transcript = ""
                        continue

                    if trans:
                        iterating_transcript += trans
                        print(iterating_transcript)
                        continue

        The generator yields both a boolean (is_last) to indicate whether it is a full utterance
        (detected by silences in audio input) and the (current/part) transcription. If the is_last boolean is true,
        then it is a full utterance determined by a silence.

        **Warning:** This method assumes that you use a model with default spectrogram/audio parameters i.e. 20ms
        audio for each stft and 50% overlap.
        """

        lookahead_context = self.danspeech_recognizer.model.context
        required_spec_frames = (lookahead_context - 1) * 2

        samples_pr_10ms = int(source.sampling_rate / 100)

        # First takes two samples pr 10ms, the rest needs 160 due to overlapping
        general_sample_requirement = samples_pr_10ms * 2 + (samples_pr_10ms * (required_spec_frames - 1))

        # First pass, we need more samples due to padding of initial conv layers
        first_sample_requirement = general_sample_requirement + (samples_pr_10ms * 15)

        data_array = []
        is_first_data = True
        is_first_pass = True
        stopper, data_getter = self.listen_in_background(source)
        self.stream_thread_stopper = stopper
        is_last = False
        output = None
        consecutive_fails = 0
        data_success = False
        # Wait 0.2 seconds before we start processing to let the background thread spawn
        time.sleep(0.2)
        while self.stream:

            # Loop for data (gets all the available data from the stream)
            while True:

                # If it is the last one in a stream, break and perform recognition no matter what
                if is_last:
                    break

                # Get all available data
                try:
                    if is_first_data:
                        is_last, data_array = data_getter()
                        is_first_data = False
                        data_success = True
                    else:
                        is_last, temp = data_getter()
                        data_array = np.concatenate((data_array, temp))
                        data_success = True
                # If this exception is thrown, then we have no available data
                except NoDataInBuffer:
                    # If it is first data and no data in buffer, then do not break but sleep.

                    # We got some data, now process
                    if data_success:
                        data_success = False
                        consecutive_fails = 0
                        break

                    # We did not get data and it was the first try, sleep for 0.4 seconds
                    if is_first_data:
                        time.sleep(0.4)
                    else:
                        consecutive_fails += 1

                    # If two fails happens in a row, we sleep for 0.3 seconds
                    if consecutive_fails == 2:
                        consecutive_fails = 0
                        time.sleep(0.3)

            # If it is the first pass, then we try to pass it
            if is_first_pass:

                # If is last and we have not performed first pass, then it should be discarded and we continue
                if is_last:
                    output = None

                # Check if we have enough frames for first pass
                elif len(data_array) >= first_sample_requirement:
                    output = self.danspeech_recognizer.streaming_transcribe(data_array,
                                                                            is_last=False,
                                                                            is_first=True)
                    # Now first pass has been performed
                    is_first_pass = False

                    # Gather new data buffer
                    data_array = []
                    is_first_data = True
            else:

                # If is last, we do not care about general sample requirement but just pass it through
                if is_last:
                    output = self.danspeech_recognizer.streaming_transcribe(data_array,
                                                                            is_last=is_last,
                                                                            is_first=False)
                    # Gather new data buffer
                    data_array = []
                    is_first_data = True

                # General case! We need some data.
                elif len(data_array) >= general_sample_requirement:
                    output = self.danspeech_recognizer.streaming_transcribe(data_array,
                                                                            is_last=is_last,
                                                                            is_first=False)

                    # Gather new data buffer
                    data_array = []
                    is_first_data = True

            # Is last should always generate output!
            if is_last and output:
                yield is_last, output

            elif output:
                yield is_last, output
                output = None

            # Reset streaminng
            if is_last:
                is_first_pass = True
                is_last = False
                output = None

    def adjust_for_speech(self, source, duration=4):
        """
        Adjusts the energy level threshold required for the :meth:`audio.Microphone` to detect
        speech in background.

        **Warning:** You need to talk after calling this method! Else, the energy level will be too low. If talking
        to adjust energy level is not an option, use :meth:`Recognizer.adjust_for_ambient_noise` instead.

        Only use if the default energy level does not match your use case.

        :param Microphone source: Source of audio.
        :param float duration: Maximum duration of adjusting the energy threshold

        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before adjusting, " \
                                          "see documentation for ``AudioSource``; are you using ``source``" \
                                          " outside of a ``with`` statement?"
        assert self.pause_threshold >= self.non_speaking_duration >= 0

        seconds_per_buffer = (source.chunk + 0.0) / source.sampling_rate
        elapsed_time = 0

        energy_levels = []
        # adjust energy threshold until a phrase starts
        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration:
                break

            buffer = source.stream.read(source.chunk)
            energy = audioop.rms(buffer, source.sampling_width)  # energy of the audio signal
            energy_levels.append(energy)

        energy_average = sum(energy_levels) / len(energy_levels)

        # Subtract some ekstra energy, since we take average
        if energy_average > 80:
            self.energy_threshold = energy_average - 80
        else:
            self.energy_threshold = energy_average

    def adjust_for_ambient_noise(self, source, duration=2):
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
        Modified for DanSpeech

        Adjusts the energy level threshold required for the :meth:`audio.Microphone` to detect
        speech in background. It is based on the energy level in the background.

        **Warning:** Do not talk while adjusting energy threshold with this method. This method generally
        sets the energy level very low. We recommend using :meth:`Recognizer.adjust_for_speech` instead.

        Only use if the default energy level does not match your use case.

        :param Microphone source: Source of audio.
        :param float duration: Maximum duration of adjusting the energy threshold

        """
        assert isinstance(source, SpeechSource), "Source must be an audio source"
        assert source.stream is not None, "Audio source must be entered before adjusting, " \
                                          "see documentation for ``AudioSource``; are you using " \
                                          "``source`` outside of a ``with`` statement?"
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


    def update_stream_parameters(self, energy_threshold=None, pause_threshold=None,
                                 phrase_threshold=None, non_speaing_duration=None):
        """
        Updates parameters for stream of audio. Only use if the default streaming from
        your microphone is working poorly.

        :param float energy_threshold: Minimum audio energy required for the stream to start detecting an utterance.
        :param float pause_threshold: Seconds of non-speaking audio before a phrase is considered complete.
        :param float phrase_threshold: Minimum seconds of speaking audio before we consider the speaking audio a phrase.
        :param float non_speaing_duration: Seconds of non-speaking audio to keep on both sides of the recording.
        """
        if energy_threshold:
            self.energy_threshold = energy_threshold
        if pause_threshold:
            self.pause_threshold = pause_threshold
        if phrase_threshold:
            self.phrase_threshold = phrase_threshold
        if non_speaing_duration:
            self.non_speaking_duration = non_speaing_duration