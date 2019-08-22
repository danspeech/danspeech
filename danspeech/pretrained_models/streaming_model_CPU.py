import torch

from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"

# ToDO: Create
def CPUStreamingRNN(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="CPUStreamingRNN.pth", origin=MODEL_PACKAGE, file_hash="0d21d9754d66d985b760ec27abd014c2", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model

