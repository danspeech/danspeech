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

# attempt to use the Python 2 modules
try:
    from urllib import urlencode
    from urllib2 import Request, urlopen, URLError, HTTPError
except ImportError:
    # use the Python 3 modules
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError


def shutil_which(pgm):
    """Python 2 compatibility: backport of ``shutil.which()`` from Python 3"""
    path = os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, pgm)
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p


def get_flac_converter():
    """Returns the absolute path of a FLAC converter executable, or raises an OSError if none can be found."""
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
            raise OSError("FLAC conversion utility not available - consider installing the FLAC command line application by running `apt-get install flac` or your operating system's equivalent")

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
    """data must be the string containing the bytes from the wav file."""
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

    def __init__(self, filepath):
        self.FILEPATH = filepath
        self.SAMPLE_RATE = None
        self.DURATION = None
        self.CHUNK = None
        self.FRAME_COUNT = None
        self.stream = None
        self.little_endian = False
        self.data = None
        self.audio_reader = None
        self.little_endian = False

    def __enter__(self):
        """
        Copyright (c) 2014-2017, Anthony Zhang <azhang9@gmail.com>

        All rights reserved.
        https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

        Modifications:
            - DanSpeech
        """
        try:
            # attempt to read the file as WAV
            self.audio_reader = wave.open(self.FILEPATH, "rb")
            # RIFF WAV is a little-endian format (most ``audioop`` operations assume that the frames are stored in little-endian form)
            self.little_endian = True
        except (wave.Error, EOFError):
            try:
                # attempt to read the file as AIFF
                self.audio_reader = aifc.open(self.FILEPATH, "rb")
                # AIFF is a big-endian format
                self.little_endian = False
            except (aifc.Error, EOFError):
                # attempt to read the file as FLAC
                if hasattr(self.FILEPATH, "read"):
                    flac_data = self.FILEPATH.read()
                else:
                    with open(self.FILEPATH, "rb") as f:
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
                    "--stdout", "--totally-silent",  # put the resulting AIFF file in stdout, and make sure it's not mixed with any program output
                    "--decode", "--force-aiff-format",  # decode the FLAC file into an AIFF file
                    "-",  # the input FLAC file contents will be given in stdin
                ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=startup_info)
                aiff_data, _ = process.communicate(flac_data)
                aiff_file = io.BytesIO(aiff_data)
                try:
                    self.audio_reader = aifc.open(aiff_file, "rb")
                except (aifc.Error, EOFError):
                    raise ValueError("Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; check if file is corrupted or in another format")
                self.little_endian = False  # AIFF is a big-endian format
        assert 1 <= self.audio_reader.getnchannels() <= 2, "Audio must be mono or stereo"
        self.SAMPLE_WIDTH = self.audio_reader.getsampwidth()

        self.SAMPLE_RATE = self.audio_reader.getframerate()

        self.CHUNK = 4096
        self.FRAME_COUNT = self.audio_reader.getnframes()
        self.DURATION = self.FRAME_COUNT / float(self.SAMPLE_RATE)
        self.stream = SpeechFile.SpeechFileStream(self.audio_reader, self.little_endian)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # only close the file if it was opened by this class in the first place (if the file was originally given as a path)
        if not hasattr(self.FILEPATH, "read"):
            self.audio_reader.close()
        self.stream = None
        self.DURATION = None

    class SpeechFileStream(object):
        def __init__(self, audio_reader, little_endian):
            # an audio file object (e.g., a `wave.Wave_read` instance)
            self.audio_reader = audio_reader
            # whether the audio data is little-endian (when working with big-endian things, we'll have to convert it to little-endian before we process it)
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


class MicroPhone(SpeechSource):
    """
    For later iteratons
    """

    def __init__(self):
        pass


class AudioData(object):
    """
    Creates a new ``AudioData`` instance, which represents mono audio data.

    The raw audio data is specified by ``frame_data``, which is a sequence of bytes representing audio samples. This is the frame data structure used by the PCM WAV format.

    The width of each sample, in bytes, is specified by ``sample_width``. Each group of ``sample_width`` bytes represents a single audio sample.

    The audio data is assumed to have a sample rate of ``sample_rate`` samples per second (Hertz).

    Usually, instances of this class are obtained from ``recognizer_instance.record`` or ``recognizer_instance.listen``, or in the callback for ``recognizer_instance.listen_in_background``, rather than instantiating them directly.
    """
    def __init__(self, frame_data, sample_rate, sample_width):
        assert sample_rate > 0, "Sample rate must be a positive integer"
        assert sample_width % 1 == 0 and 1 <= sample_width <= 4, "Sample width must be between 1 and 4 inclusive"
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = int(sample_width)

    def get_segment(self, start_ms=None, end_ms=None):
        """
        Returns a new ``AudioData`` instance, trimmed to a given time interval. In other words, an ``AudioData`` instance with the same audio data except starting at ``start_ms`` milliseconds in and ending ``end_ms`` milliseconds in.

        If not specified, ``start_ms`` defaults to the beginning of the audio, and ``end_ms`` defaults to the end.
        """
        assert start_ms is None or start_ms >= 0, "``start_ms`` must be a non-negative number"
        assert end_ms is None or end_ms >= (0 if start_ms is None else start_ms), " ``end_ms`` must be a non-negative number greater or equal to ``start_ms``"
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
        Returns a byte string representing the raw frame data for the audio represented by the ``AudioData`` instance.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        Writing these bytes directly to a file results in a valid `RAW/PCM audio file <https://en.wikipedia.org/wiki/Raw_audio_format>`__.
        """
        assert convert_rate is None or convert_rate > 0, "Sample rate to convert to must be a positive integer"
        assert convert_width is None or (convert_width % 1 == 0 and 1 <= convert_width <= 4), "Sample width to convert to must be between 1 and 4 inclusive"

        raw_data = self.frame_data

        # make sure unsigned 8-bit audio (which uses unsigned samples) is handled like higher sample width audio (which uses signed samples)
        if self.sample_width == 1:
            raw_data = audioop.bias(raw_data, 1, -128)  # subtract 128 from every sample to make them act like signed samples

        # resample audio at the desired rate if specified
        if convert_rate is not None and self.sample_rate != convert_rate:
            raw_data, _ = audioop.ratecv(raw_data, self.sample_width, 1, self.sample_rate, convert_rate, None)

        # convert samples to desired sample width if specified
        if convert_width is not None and self.sample_width != convert_width:
            if convert_width == 3:  # we're converting the audio into 24-bit (workaround for https://bugs.python.org/issue12866)
                raw_data = audioop.lin2lin(raw_data, self.sample_width, 4)  # convert audio into 32-bit first, which is always supported
                try: audioop.bias(b"", 3, 0)  # test whether 24-bit audio is supported (for example, ``audioop`` in Python 3.3 and below don't support sample width 3, while Python 3.4+ do)
                except audioop.error:  # this version of audioop doesn't support 24-bit audio (probably Python 3.3 or less)
                    raw_data = b"".join(raw_data[i + 1:i + 4] for i in range(0, len(raw_data), 4))  # since we're in little endian, we discard the first byte from each 32-bit sample to get a 24-bit sample
                else:  # 24-bit audio fully supported, we don't need to shim anything
                    raw_data = audioop.lin2lin(raw_data, self.sample_width, convert_width)
            else:
                raw_data = audioop.lin2lin(raw_data, self.sample_width, convert_width)

        # if the output is 8-bit audio with unsigned samples, convert the samples we've been treating as signed to unsigned again
        if convert_width == 1:
            raw_data = audioop.bias(raw_data, 1, 128)  # add 128 to every sample to make them act like unsigned samples again

        return raw_data

    def get_wav_data(self, convert_rate=None, convert_width=None):
        """
        Returns a byte string representing the contents of a WAV file containing the audio represented by the ``AudioData`` instance.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        Writing these bytes directly to a file results in a valid `WAV file <https://en.wikipedia.org/wiki/WAV>`__.
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
        raw_data = self.get_raw_data(convert_rate, convert_width)
        sample_width = self.sample_width if convert_width is None else convert_width
        return _wav2array(1, sample_width, raw_data).squeeze().astype(float)

    def get_aiff_data(self, convert_rate=None, convert_width=None):
        """
        Returns a byte string representing the contents of an AIFF-C file containing the audio represented by the ``AudioData`` instance.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        Writing these bytes directly to a file results in a valid `AIFF-C file <https://en.wikipedia.org/wiki/Audio_Interchange_File_Format>`__.
        """
        raw_data = self.get_raw_data(convert_rate, convert_width)
        sample_rate = self.sample_rate if convert_rate is None else convert_rate
        sample_width = self.sample_width if convert_width is None else convert_width

        # the AIFF format is big-endian, so we need to covnert the little-endian raw data to big-endian
        if hasattr(audioop, "byteswap"):  # ``audioop.byteswap`` was only added in Python 3.4
            raw_data = audioop.byteswap(raw_data, sample_width)
        else:  # manually reverse the bytes of each sample, which is slower but works well enough as a fallback
            raw_data = raw_data[sample_width - 1::-1] + b"".join(raw_data[i + sample_width:i:-1] for i in range(sample_width - 1, len(raw_data), sample_width))

        # generate the AIFF-C file contents
        with io.BytesIO() as aiff_file:
            aiff_writer = aifc.open(aiff_file, "wb")
            try:  # note that we can't use context manager, since that was only added in Python 3.4
                aiff_writer.setframerate(sample_rate)
                aiff_writer.setsampwidth(sample_width)
                aiff_writer.setnchannels(1)
                aiff_writer.writeframes(raw_data)
                aiff_data = aiff_file.getvalue()
            finally:  # make sure resources are cleaned up
                aiff_writer.close()
        return aiff_data

    def get_flac_data(self, convert_rate=None, convert_width=None):
        """
        Returns a byte string representing the contents of a FLAC file containing the audio represented by the ``AudioData`` instance.

        Note that 32-bit FLAC is not supported. If the audio data is 32-bit and ``convert_width`` is not specified, then the resulting FLAC will be a 24-bit FLAC.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        Writing these bytes directly to a file results in a valid `FLAC file <https://en.wikipedia.org/wiki/FLAC>`__.
        """
        assert convert_width is None or (convert_width % 1 == 0 and 1 <= convert_width <= 3), "Sample width to convert to must be between 1 and 3 inclusive"

        if self.sample_width > 3 and convert_width is None:  # resulting WAV data would be 32-bit, which is not convertable to FLAC using our encoder
            convert_width = 3  # the largest supported sample width is 24-bit, so we'll limit the sample width to that

        # run the FLAC converter with the WAV data to get the FLAC data
        wav_data = self.get_wav_data(convert_rate, convert_width)
        flac_converter = get_flac_converter()
        if os.name == "nt":  # on Windows, specify that the process is to be started without showing a console window
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # specify that the wShowWindow field of `startup_info` contains a value
            startup_info.wShowWindow = subprocess.SW_HIDE  # specify that the console window should be hidden
        else:
            startup_info = None  # default startupinfo
        process = subprocess.Popen([
            flac_converter,
            "--stdout", "--totally-silent",  # put the resulting FLAC file in stdout, and make sure it's not mixed with any program output
            "--best",  # highest level of compression available
            "-",  # the input FLAC file contents will be given in stdin
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=startup_info)
        flac_data, stderr = process.communicate(wav_data)
        return flac_data

"""
Local dev tests
Should be removed
"""
if __name__ == '__main__':
    file_path = "../example_files/u0013002.wav"
    with SpeechFile(filepath=file_path, sampling_rate=16000) as source:
        pass

