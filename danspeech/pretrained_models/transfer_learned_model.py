from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


MODEL_PACKAGE = 'https://github.com/danspeech/danspeech/releases/download/v0.01-alpha/TransferLearned.pth'


def TransferLearned(cache_dir=None):
    """
    The Librispeech English model adapted to Danish while keeping the conv layers and the lowest/first RNN layer frozen

    This model performs better than the DanSpeechPrimary model on noisy data.

    2 Conv layers

    5 RNN Layers with 800 hidden units

    param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
    recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Pretrained DeepSpeech (Transfer learned from English) model
    :rtype: ``danspeech.deepspeech.model.DeepSpeech``
    """
    model_path = get_model(model_name="TransferLearned.pth", origin=MODEL_PACKAGE,
                           file_hash="d19b9d7dc976bffbc9225e0f80ecacbf", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
