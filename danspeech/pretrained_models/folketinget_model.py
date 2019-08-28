from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = 'https://github.com/danspeech/danspeech/releases/download/v0.01-alpha/Folketinget.pth'


def Folketinget(cache_dir=None):
    """
    The deepest and best performing DanSpeech model finetuned to data from Folketinget.

    3 Conv layers

    9 RNN Layers with 1200 hidden units

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Pretrained DeepSpeech (Folketinget tuned) model.
    :rtype: ``danspeech.deepspeech.model.DeepSpeech``
    """
    model_path = get_model(model_name="Folketinget.pth", origin=MODEL_PACKAGE,
                           file_hash="9523d5744ad4ff5ffc8519393350cc91", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
