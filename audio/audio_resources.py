from abc import ABC, abstractmethod


class SpeechSource(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def (self):
        pass



class SpeechFile(SpeechSource):

    def __init__(self, filename, sampling_rate):
        self.filename_or_fileobject =
        self.stream = None
        self.DURATION = None

        self.audio_reader = None
        self.little_endian = False
        self.SAMPLE_RATE = None
        self.CHUNK = None
        self.FRAME_COUNT = None



class MicroPhone(SpeechSource):

    def __init__(self):