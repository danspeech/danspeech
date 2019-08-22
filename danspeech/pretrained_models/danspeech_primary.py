from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"


def DanSpeechPrimary(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="DanSpeechPrimary.pth", origin=MODEL_PACKAGE, file_hash="d169900e0781047f0a19a6efbef353ee", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
