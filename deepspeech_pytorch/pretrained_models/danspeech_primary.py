import torch

from deepspeech_pytorch.model import DeepSpeech
from .data_utils import get_model

MODEL_PACKAGE = "toDo"


def DanSpeechPrimary():
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="danspeech.pth", origin=MODEL_PACKAGE, file_hash="")
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    return model
