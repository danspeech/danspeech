import torch

from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"

def DanSpeechFolketinget(cache_dir=None):
    """
    Instantiates the finetunded folketinget DanSpeech model

    :return: Pretrained DeepSpeech model
    """

    model_path = get_model(model_name="folketinget_finetuned_6.pth", origin=MODEL_PACKAGE, file_hash="a1beed5439a508b9d7a3aa980d4f89ed", cache_dir=cache_dir)
    package = torch.load(model_path, map_location=lambda storage, loc: storage)

    # ToDO: Fix the need of conv layers specificer... Should be deprecated when new models are trained
    model = DeepSpeech.load_model_package(package, conv_layers=3)
    return model