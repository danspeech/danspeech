import torch

from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"


def DanSpeechPrimary(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="danspeech_primary.pth", origin=MODEL_PACKAGE, file_hash="2dfa5b8bdee1970d20c6594014334436", cache_dir=cache_dir)
    package = torch.load(model_path, map_location=lambda storage, loc: storage)

    # ToDO: Fix the need of conv layers specificer... Should be deprecated when new models are trained
    model = DeepSpeech.load_model_package(package, conv_layers=3)
    return model
