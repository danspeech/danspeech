import torch

from deepspeech_pytorch.model import DeepSpeech
from .data_utils import get_model

MODEL_PACKAGE = 'https://drive.google.com/open?id=1jFTZZNISA-qiJfLjfZaM21LY4niaYU_M'

WEIGHTS_PATH_NO_TOP = "ToDO not"


# ToDO: Top weights option
def RNN3():
    """
    Instantiates the most complex DanSpeech model with a lot of parameters


    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="3RNN.pth", origin=MODEL_PACKAGE, file_hash="toDo")
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)
    return model


