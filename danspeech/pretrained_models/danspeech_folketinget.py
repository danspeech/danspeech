import torch

from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"

def DanSpeechFolketinget(cache_dir=None):
    """
    Instantiates the finetunded folketinget DanSpeech model

    :return: Pretrained DeepSpeech model
    """

    model_path = get_model(model_name="folketinget_finetuned_4.pth", origin=MODEL_PACKAGE, file_hash="452fb02528b7a97ee0ba4ad9ab2fd31d", cache_dir=cache_dir)
    package = torch.load(model_path, map_location=lambda storage, loc: storage)

    # ToDO: Fix the need of conv layers specificer... Should be deprecated when new models are trained
    model = DeepSpeech.load_model_package(package, conv_layers=3)
    return model