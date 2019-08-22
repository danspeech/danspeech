import torch

from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"

# ToDO: Create
def GPUStreamingRNN(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="unlookahead_test_large.pth", origin=MODEL_PACKAGE, file_hash="79f0e495335a76e7da1455a56e4f6a6e", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
