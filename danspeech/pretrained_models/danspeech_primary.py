from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


# ToDo: Add model package link for release
MODEL_PACKAGE = "toDo"


def DanSpeechPrimary(cache_dir=None):
    """
    Deepest and best performing DanSpeech model.

    3 Conv layers
    9 RNN Layers with 1200 hidden units

    WARNING: This model is really slow, so we suggest you use it with a GPU.

    :return: Pretrained DeepSpeech (Best Performing) model
    """
    model_path = get_model(model_name="DanSpeechPrimary.pth", origin=MODEL_PACKAGE,
                           file_hash="5bd08282d442e990c37481d5c61cf93c", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
