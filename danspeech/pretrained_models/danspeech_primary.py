import torch

from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"


def DanSpeechPrimary(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="danspeech.pth", origin=MODEL_PACKAGE, file_hash="", cache_dir=cache_dir)
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    return model
