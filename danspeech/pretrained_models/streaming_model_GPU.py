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
    model_path = get_model(model_name="GPUStreamingRNN.pth", origin=MODEL_PACKAGE, file_hash="065ff4f4699c96ec7fd613a193bbcc17", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
