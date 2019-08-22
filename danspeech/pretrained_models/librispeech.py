from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"


def EnglishLibrispeech(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="Librispeech.pth", origin=MODEL_PACKAGE, file_hash="56630094905e7308f42ae0f82421440b", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
