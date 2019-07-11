import torch

from deepspeech_pytorch.model import DeepSpeech
from .data_utils import get_model

MODEL_PACKAGE = 'https://github.com/Rasmusafj/models_development/raw/master/400units.pth'


def Units400():
    """
    Instantiates model with 2 conv layers and 5 rnn layers each with 400 units



    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="400_units.pth", origin=MODEL_PACKAGE, file_hash="1bb9c5b6bb8259193b5ac7e2e0490cd7")
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    return model


