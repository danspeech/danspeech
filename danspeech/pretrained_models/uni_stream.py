import torch

from danspeech.deepspeech.model import DeepSpeechStreamInference
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"


def StreamingRNN(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="uni_lookahead_test.pth", origin=MODEL_PACKAGE, file_hash="288173d11be68865d1dfcc8c0319d4c4", cache_dir=cache_dir)
    model = DeepSpeechStreamInference.load_model(model_path)
    return model

