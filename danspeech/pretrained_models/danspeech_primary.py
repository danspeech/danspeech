from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"


def DanSpeechPrimary(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="DanSpeechPrimary.pth", origin=MODEL_PACKAGE, file_hash="5bd08282d442e990c37481d5c61cf93c", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
