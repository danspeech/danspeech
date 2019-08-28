from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


MODEL_PACKAGE = "https://github.com/danspeech/danspeech/releases/download/v0.01-alpha/DanSpeechPrimary.pth"


def DanSpeechPrimary(cache_dir=None):
    """
    Deepest and best performing DanSpeech model.

    3 Conv layers

    9 RNN Layers with 1200 hidden units

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Pretrained DeepSpeech (Best Performing) model.
    :rtype: ``danspeech.deepspeech.model.DeepSpeech``
    """
    model_path = get_model(model_name="DanSpeechPrimary.pth", origin=MODEL_PACKAGE,
                           file_hash="5bd08282d442e990c37481d5c61cf93c", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
