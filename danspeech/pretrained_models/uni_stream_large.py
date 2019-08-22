import torch

from danspeech.deepspeech.model import DeepSpeechStreamInference
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"


def StreamingRNNLarge(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="unlookahead_test_large.pth", origin=MODEL_PACKAGE, file_hash="79f0e495335a76e7da1455a56e4f6a6e", cache_dir=cache_dir)
    model = DeepSpeechStreamInference.load_model(model_path)
    return model
