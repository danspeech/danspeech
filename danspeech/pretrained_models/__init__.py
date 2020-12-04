from .danspeech_primary import DanSpeechPrimary
from .test_model import TestModel
from .baseline_model import Baseline
from .streaming_model_CPU import CPUStreamingRNN
from .streaming_model_GPU import GPUStreamingRNN
from .folketinget_model import Folketinget
from .transfer_learned_model import TransferLearned
from .librispeech import EnglishLibrispeech
from .custom_model import CustomModel


def get_model_from_string(model_name):
    if model_name == 'DanSpeechPrimary':
        return DanSpeechPrimary()
    elif model_name == 'TestModel':
        return TestModel()
    elif model_name == 'Baseline':
        return Baseline()
    elif model_name == 'CPUStreamingRNN':
        return CPUStreamingRNN()
    elif model_name == 'GPUStreamingRNN':
        return CPUStreamingRNN()
    elif model_name == 'Folketinget':
        return Folketinget()
    elif model_name == 'TransferLearned':
        return TransferLearned()
    elif model_name == 'EnglishLibrispeech':
        return EnglishLibrispeech()
    else:
        return None
