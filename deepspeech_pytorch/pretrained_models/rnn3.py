import torch

from deepspeech_pytorch.model import DeepSpeech
from .data_utils import get_model

MODEL_PACKAGE = 'ToDo'


def Rnn3():
    """
    Instantiates model with 2 conv layers and 3 rnn layers each with 800 units

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="3RNN.pth", origin=MODEL_PACKAGE, file_hash="toDo")
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    return model


