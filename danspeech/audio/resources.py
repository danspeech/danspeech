import warnings

import numpy as np
import os
import stat
import sys
import wave
import aifc
import platform
import subprocess
import audioop
import io

from abc import ABC, abstractmethod
import scipy.io.wavfile as wav


class SamplingRateWarning(Warning):
    pass


def load_audio(path, duration=None, offset=None):
    """
    Loads a sound file.

    Supported formats are WAV, AIFF, FLAC.

    :param str path: Path to sound file
    :param float duration: Duration in seconds of how much to use. If duration is not specified,
        then it will record until there is no more audio input.
    :param float offset: Where to start in seconds in the clip.
    :return: Input array ready for speech recognition.
    :rtype: ``numpy.array``
    """

    with SpeechFile(filepath=path) as source:
        frames_bytes = io.BytesIO()
        seconds_per_buffer = (source.chunk + 0.0) / source.sampling_rate
        elapsed_time = 0
        offset_time = 0
        offset_reached = False
        while True:  # loop for the total number of chunks needed
            if offset and not offset_reached:
                offset_time += seconds_per_buffer
                if offset_time > offset:
                    offset_reached = True

            buffer = source.stream.read(source.chunk)
            if len(buffer) == 0:
                break

            if offset_reached or not offset:
                elapsed_time += seconds_per_buffer
                if duration and elapsed_time > duration:
                    break

                frames_bytes.write(buffer)

        frame_data = frames_bytes.getvalue()
        frames_bytes.close()
        return AudioData(frame_data, source.sampling_rate, source.sampling_width).get_array_data()


def load_audio_wavPCM(path):
    """
    Fast load of wav.

    This works well if you are certain that your wav files are PCM encoded.

    :param str path: Path to wave file.
    :return: Input array ready for speech recognition.
    :rtype: ``numpy.array``
    """
    _, sound = wav.read(path)

    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average

    return sound.astype(float)


def shutil_which(pgm):
    """
    Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
    Python 2 compatibility: backport of ``shutil.which()`` from Python 3
    """
    path = os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, pgm)
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p


def get_flac_converter():
    """
    Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

    :return: the absolute path of a FLAC converter executable, or raises an OSError if none can be found.
    """
    # check for installed version first
    flac_converter = shutil_which("flac")
    if flac_converter is None:
        # flac utility is not installed
        base_path = os.path.dirname(os.path.abspath(__file__))

        # directory of the current module file, where all the FLAC bundled binaries are stored
        system, machine = platform.system(), platform.machine()
        if system == "Windows" and machine in {"i686", "i786", "x86", "x86_64", "AMD64"}:
            flac_converter = os.path.join(base_path, "flac-win32.exe")
        elif system == "Darwin" and machine in {"i686", "i786", "x86", "x86_64", "AMD64"}:
            flac_converter = os.path.join(base_path, "flac-mac")
        elif system == "Linux" and machine in {"i686", "i786", "x86"}:
            flac_converter = os.path.join(base_path, "flac-linux-x86")
        elif system == "Linux" and machine in {"x86_64", "AMD64"}:
            flac_converter = os.path.join(base_path, "flac-linux-x86_64")
        else:
            # no FLAC converter available
            raise OSError(
                "FLAC conversion utility not available - consider installing the FLAC command line application by "
                "running `apt-get install flac` or your operating system's equivalent")

    # mark FLAC converter as executable if possible
    try:
        # handle known issue when running on docker:
        # run executable right after chmod() may result in OSError "Text file busy"
        # fix: flush FS with sync
        if not os.access(flac_converter, os.X_OK):
            stat_info = os.stat(flac_converter)
            os.chmod(flac_converter, stat_info.st_mode | stat.S_IEXEC)
            if 'Linux' in platform.system():
                os.sync() if sys.version_info >= (3, 3) else os.system('sync')

    except OSError:
        pass

    return flac_converter


def _wav2array(nchannels, sampwidth, data):
    """
    Source: https://github.com/WarrenWeckesser/wavio

    Converts bytestring to array.

    :param nchannels: Number of channels in wav
    :param sampwidth: Sample width
    :param data: data must be the string containing the bytes from the wav file.
    :return: Numpy array containing the speech data
    """
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


class SpeechSource(ABC):

    @abstractmethod
    def __init__(self):
        pass


class SpeechFile(SpeechSource):
    """
    Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
    Modified for DanSpeech

    This is a checker of the speech file which also streams the input.

    This class is wrapped in danspeech.audio.resources.load_audio and should hence not be directly called.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.sampling_rate = 16000
        self.duration = None
        self.chunk = None
        self.frame_count = None
        self.stream = None
        self.little_endian = False
        self.audio_reader = None
        self.sampling_width = None

    def __enter__(self):

        try:
            # attempt to read the file as WAV
            self.audio_reader = wave.open(self.filepath, "rb")
            # RIFF WAV is a little-endian format (most ``audioop`` operations assume that the
            # frames are stored in little-endian form)
            self.little_endian = True
        except (wave.Error, EOFError):
            try:
                # attempt to read the file as AIFF
                self.audio_reader = aifc.open(self.filepath, "rb")
                # AIFF is a big-endian format
                self.little_endian = False
            except (aifc.Error, EOFError):
                # attempt to read the file as FLAC
                if hasattr(self.filepath, "read"):
                    flac_data = self.filepath.read()
                else:
                    with open(self.filepath, "rb") as f:
                        flac_data = f.read()

                # run the FLAC converter with the FLAC data to get the AIFF data
                flac_converter = get_flac_converter()
                # on Windows, specify that the process is to be started without showing a console window
                if os.name == "nt":
                    startup_info = subprocess.STARTUPINFO()
                    # specify that the wShowWindow field of `startup_info` contains a value
                    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    # specify that the console window should be hidden
                    startup_info.wShowWindow = subprocess.SW_HIDE
                else:
                    # default startupinfo
                    startup_info = None
                process = subprocess.Popen([
                    flac_converter,
                    "--stdout", "--totally-silent",
                    # put the resulting AIFF file in stdout, and make sure it's not mixed with any program output
                    "--decode", "--force-aiff-format",  # decode the FLAC file into an AIFF file
                    "-",  # the input FLAC file contents will be given in stdin
                ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=startup_info)
                aiff_data, _ = process.communicate(flac_data)
                aiff_file = io.BytesIO(aiff_data)
                try:
                    self.audio_reader = aifc.open(aiff_file, "rb")
                except (aifc.Error, EOFError):
                    raise ValueError(
                        "Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; "
                        "check if file is corrupted or in another format")
                self.little_endian = False  # AIFF is a big-endian format

        assert 1 <= self.audio_reader.getnchannels() <= 2, "Audio must be mono or stereo"
        self.sampling_width = self.audio_reader.getsampwidth()

        if self.sampling_rate != self.audio_reader.getframerate():
            warnings.warn(
                "Specified file {0} sampling rate. DanSpeech currently only supports 16000 sampling rate. Will "
                "resample to 16000 sampling rate".format(self.audio_reader.getframerate()),
                SamplingRateWarning)

        self.chunk = 4096
        self.frame_count = self.audio_reader.getnframes()
        self.duration = self.frame_count / float(self.sampling_rate)
        self.stream = SpeechFile.SpeechFileStream(self.audio_reader, self.little_endian)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # only close the file if it was opened by this class in the first place
        # (if the file was originally given as a path)
        if not hasattr(self.filepath, "read"):
            self.audio_reader.close()
        self.stream = None
        self.duration = None

    class SpeechFileStream(object):
        def __init__(self, audio_reader, little_endian):
            # an audio file object (e.g., a `wave.Wave_read` instance)
            self.audio_reader = audio_reader
            # whether the audio data is little-endian (when working with big-endian things,
            # we'll have to convert it to little-endian before we process it)
            self.little_endian = little_endian

        def read(self, size=-1):
            # workaround for https://bugs.python.org/issue24608
            buffer = self.audio_reader.readframes(self.audio_reader.getnframes() if size == -1 else size)
            if not isinstance(buffer, bytes):
                buffer = b""

            sample_width = self.audio_reader.getsampwidth()
            # big endian format, convert to little endian on the fly
            if not self.little_endian:
                # ``audioop.byteswap`` was only added in Python 3.4 (incidentally, that also means that we don't
                # need to worry about 24-bit audio being unsupported, since Python 3.4+ always has that functionality)
                if hasattr(audioop, "byteswap"):
                    buffer = audioop.byteswap(buffer, sample_width)
                else:
                    # manually reverse the bytes of each sample, which is slower but works well enough as a fallback
                    buffer = buffer[sample_width - 1::-1] + b"".join(
                        buffer[i + sample_width:i:-1] for i in range(sample_width - 1, len(buffer), sample_width))

            # convert stereo audio data to mono
            if self.audio_reader.getnchannels() != 1:
                buffer = audioop.tomono(buffer, sample_width, 1, 1)
            return buffer


def get_pyaudio():
    """
    Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

    Imports the pyaudio module and checks its version. Throws exceptions if pyaudio can't be found or a
    wrong version is installed
    """
    try:
        import pyaudio
    except ImportError:
        raise AttributeError("Could not find PyAudio; check installation")
    from distutils.version import LooseVersion
    if LooseVersion(pyaudio.__version__) < LooseVersion("0.2.11"):
        raise AttributeError("PyAudio 0.2.11 or later is required (found version {})".format(pyaudio.__version__))
    return pyaudio


class Microphone(SpeechSource):
    """
    Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

    Modified for DanSpeech

    **Warning:** Requires PyAudio.

    Creates a Microphone instance, which represents the a microphone on the computer.

    The microphone needs a device index, or else it will try to use the default microphone of the system.

    Sampling rate should always be 16000, if the microphone should work with DanSpeech models.

    :param int device_index: The device index of your microphone. Use :meth:`Microphone.list_microphone_names` to
        find the available input sources and choose the appropriate one.
    :param int sampling_rate: Should always be 16000 unless you configured audio configuration of a
        your own trained danspeech model.
    :param int chunk_size: Avoid changing chunk size unless it is strictly neccessary.
        WARNING: Will possibly break microphone streaming with DanSpeech models.

    :Example:

         .. code-block:: python

            from danspeech import Recognizer {}
            from danspeech.pretrained_models import TestModel
            from danspeech.audio.resources import Microphone

            # Get a list of microphones found by PyAudio
            mic_list = Microphone.list_microphone_names()
            mic_list_with_numbers = list(zip(range(len(mic_list)), mic_list))
            print("Available microphones: {0}".format(mic_list_with_numbers))

            # Choose the microphone
            mic_number = input("Pick the number of the microphone you would like to use: ")

            # Init a microphone object
            m = Microphone(sampling_rate=16000, device_index=int(mic_number))

            # Init a DanSpeech model and create a Recognizer instance
            model = TestModel()
            recognizer = Recognizer(model=model)

            print("Speek a lot to adjust silence detection from microphone...")
            with m as source:
                recognizer.adjust_for_speech(source, duration=5)

            # Enable streaming
            recognizer.enable_streaming()

            # Create the streaming generator which runs a background thread listening to the microphone stream
            generator = recognizer.streaming(source=m)

            # The below code runs for a long time. The generator returns transcriptions of spoken speech from your microphone.
            print("Speak")
            for i in range(100000):
                trans = next(generator)
                print(trans)
    """

    def __init__(self, device_index=None, sampling_rate=16000, chunk_size=1024):
        assert device_index is None or isinstance(device_index, int), "Device index must be None or an integer"
        assert sampling_rate is None or (
                isinstance(sampling_rate, int) and sampling_rate > 0), "Sample rate must be None or a positive integer"
        assert isinstance(chunk_size, int) and chunk_size > 0, "Chunk size must be a positive integer"

        try:
            # set up PyAudio
            self.pyaudio_module = get_pyaudio()
            audio = self.pyaudio_module.PyAudio()
            try:
                count = audio.get_device_count()  # obtain device count
                if device_index is not None:  # ensure device index is in range
                    assert 0 <= device_index < count, "Device index out of range ({} devices available; " \
                                                      "device index should be between 0 and {} inclusive)".format(
                        count, count - 1)

                # automatically set the sample rate to the hardware's default sample rate if not specified
                if sampling_rate is None:
                    device_info = audio.get_device_info_by_index(
                        device_index) if device_index is not None else audio.get_default_input_device_info()
                    assert isinstance(device_info.get("defaultSampleRate"), (float, int)) and device_info[
                        "defaultSampleRate"] > 0, "Invalid device info returned from PyAudio: {}".format(device_info)
                    sampling_rate = int(device_info["defaultSampleRate"])
            finally:
                audio.terminate()

            self.device_index = device_index
            self.format = self.pyaudio_module.paInt16  # 16-bit int sampling
            self.sampling_width = self.pyaudio_module.get_sample_size(self.format)  # size of each sample
            self.sampling_rate = sampling_rate  # sampling rate in Hertz
            self.chunk = chunk_size  # number of frames stored in each buffer

            self.audio = None
            self.stream = None

        except AttributeError as e:
            warnings.warn("PyAudio not installed. You will not be able to use microphone", e)

    @staticmethod
    def list_microphone_names():
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

        Find all available input sources.

        The index of each microphone's name in the returned list is the same as its device index when creating
        a Microphone instance - if you want to use the microphone at index 3
        in the returned list, use ``Microphone(device_index=3)``.

        **Warning:** Will also show sources that are not actually microphones, which will result in an error. Try
        another one, that sounds plausible.

        :return: A list of the names of all available microphones.
        :rtype: list
        """

        audio = get_pyaudio().PyAudio()
        try:
            result = []
            for i in range(audio.get_device_count()):
                device_info = audio.get_device_info_by_index(i)
                result.append(device_info.get("name"))
        finally:
            audio.terminate()
        return result

    def __enter__(self):
        assert self.stream is None, "This audio source is already inside a context manager"
        self.audio = self.pyaudio_module.PyAudio()
        try:
            self.stream = Microphone.MicrophoneStream(
                self.audio.open(
                    input_device_index=self.device_index, channels=1, format=self.format,
                    rate=self.sampling_rate, frames_per_buffer=self.chunk, input=True,
                )
            )
        except Exception:
            self.audio.terminate()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.audio.terminate()

    class MicrophoneStream(object):
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

        Used to stream microphone. Will be called in Recognizer.

        """
        def __init__(self, pyaudio_stream):
            self.pyaudio_stream = pyaudio_stream

        def read(self, size):
            return self.pyaudio_stream.read(size, exception_on_overflow=False)

        def close(self):
            try:
                # sometimes, if the stream isn't stopped, closing the stream throws an exception
                if not self.pyaudio_stream.is_stopped():
                    self.pyaudio_stream.stop_stream()
            finally:
                self.pyaudio_stream.close()


class AudioData(object):
    """
    Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
    Modified for DanSpeech

    AudioData represents mono data as a bytestring.

    To get array data for DanSpeech models, use get_array_data.

    If you need to write the file to a .wav file, use get_wav_data

    WARNING: Using sample rate other than 16000 will not work with DanSpeech models.
    """

    def __init__(self, frame_data, sample_rate, sample_width):
        assert sample_rate > 0, "Sample rate must be a positive integer"
        assert sample_width % 1 == 0 and 1 <= sample_width <= 4, "Sample width must be between 1 and 4 inclusive"
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = int(sample_width)

    def get_segment(self, start_ms=None, end_ms=None):
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py
        Modified for DanSpeech

        Used if you need a specific segment of the audio data.

        :param start_ms: Start of trimmed audio
        :param end_ms: End of Trimmed audio
        :return: Returns a new AudioData instance, trimmed to a given time interval.
        """

        #  "``start_ms`` must be a non-negative number"
        assert start_ms is None or start_ms >= 0

        # ``end_ms`` must be a non-negative number greater or equal to ``start_ms``"
        assert end_ms is None or end_ms >= (0 if start_ms is None else start_ms)
        if start_ms is None:
            start_byte = 0
        else:
            start_byte = int((start_ms * self.sample_rate * self.sample_width) // 1000)
        if end_ms is None:
            end_byte = len(self.frame_data)
        else:
            end_byte = int((end_ms * self.sample_rate * self.sample_width) // 1000)
        return AudioData(self.frame_data[start_byte:end_byte], self.sample_rate, self.sample_width)

    def get_raw_data(self, convert_rate=None, convert_width=None):
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

        Writing these bytes directly to a file results in a valid
        `RAW/PCM audio file <https://en.wikipedia.org/wiki/Raw_audio_format>`__.


        :param convert_rate: Specify to convert the data into a new sample_rate
        :param convert_width: Specify to convert data into a new width
        :return: A byte string representing the raw frame data for the audio
        represented by the AudioData instance.
        """
        assert convert_rate is None or convert_rate > 0, "Sample rate to convert to must be a positive integer"
        assert convert_width is None or (convert_width % 1 == 0 and 1 <= convert_width <= 4), \
            "Sample width to convert to must be between 1 and 4 inclusive"

        raw_data = self.frame_data

        # make sure unsigned 8-bit audio (which uses unsigned samples) is handled
        # like higher sample width audio (which uses signed samples)
        if self.sample_width == 1:
            raw_data = audioop.bias(raw_data, 1,
                                    -128)  # subtract 128 from every sample to make them act like signed samples

        # resample audio at the desired rate if specified
        if convert_rate is not None and self.sample_rate != convert_rate:
            raw_data, _ = audioop.ratecv(raw_data, self.sample_width, 1, self.sample_rate, convert_rate, None)

        # convert samples to desired sample width if specified
        if convert_width is not None and self.sample_width != convert_width:
            # we're converting the audio into 24-bit (workaround for https://bugs.python.org/issue12866)
            if convert_width == 3:
                raw_data = audioop.lin2lin(raw_data, self.sample_width,
                                           4)  # convert audio into 32-bit first, which is always supported
                try:
                    # test whether 24-bit audio is supported (for example, ``audioop`` in Python 3.3
                    # and below don't support sample width 3, while Python 3.4+ do)
                    audioop.bias(b"", 3, 0)

                # this version of audioop doesn't support 24-bit audio (probably Python 3.3 or less)
                except audioop.error:
                    # since we're in little endian, we discard the first byte from each 32-bit sample
                    # to get a 24-bit sample
                    raw_data = b"".join(raw_data[i + 1:i + 4] for i in range(0, len(raw_data), 4))
                else:  # 24-bit audio fully supported, we don't need to shim anything
                    raw_data = audioop.lin2lin(raw_data, self.sample_width, convert_width)
            else:
                raw_data = audioop.lin2lin(raw_data, self.sample_width, convert_width)

        # if the output is 8-bit audio with unsigned samples, convert the samples
        # we've been treating as signed to unsigned again
        if convert_width == 1:
            raw_data = audioop.bias(raw_data, 1,
                                    128)  # add 128 to every sample to make them act like unsigned samples again

        return raw_data

    def get_wav_data(self, convert_rate=None, convert_width=None):
        """
        Source: https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

        Writing these bytes directly to a file results in a valid `WAV file <https://en.wikipedia.org/wiki/WAV>`__.

        :param convert_rate: Specify to convert the data into a new sample_rate
        :param convert_width: Specify to convert data into a new width
        :return: A byte string representing the contents of a WAV file containing
        the audio represented by the AudioData instance.
        """

        raw_data = self.get_raw_data(convert_rate, convert_width)
        sample_rate = self.sample_rate if convert_rate is None else convert_rate
        sample_width = self.sample_width if convert_width is None else convert_width

        # generate the WAV file contents
        with io.BytesIO() as wav_file:
            wav_writer = wave.open(wav_file, "wb")
            try:  # note that we can't use context manager, since that was only added in Python 3.4
                wav_writer.setframerate(sample_rate)
                wav_writer.setsampwidth(sample_width)
                wav_writer.setnchannels(1)
                wav_writer.writeframes(raw_data)
                wav_data = wav_file.getvalue()
            finally:  # make sure resources are cleaned up
                wav_writer.close()
        return wav_data

    def get_array_data(self, convert_rate=None, convert_width=None):
        """
        Get data as numpy array from an AudioData instance.

        :param convert_rate: Specify to convert the data into a new sample_rate
        :param convert_width: Specify to convert data into a new width
        :return:
        """
        raw_data = self.get_raw_data(convert_rate, convert_width)
        sample_width = self.sample_width if convert_width is None else convert_width
        return _wav2array(1, sample_width, raw_data).squeeze().astype(float)
