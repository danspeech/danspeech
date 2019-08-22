from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


# ToDo: Add model package link for release
MODEL_PACKAGE = "toDo"


def EnglishLibrispeech(cache_dir=None):
    """
    English trained model on the Librispeech corpus.

    2 Conv layers
    5 RNN Layers with 800 hidden units

    :return: Pretrained DeepSpeech (English speech recognition) model
    """
    model_path = get_model(model_name="Librispeech.pth", origin=MODEL_PACKAGE,
                           file_hash="56630094905e7308f42ae0f82421440b", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
