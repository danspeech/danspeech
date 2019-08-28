from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


MODEL_PACKAGE = 'https://github.com/danspeech/danspeech/releases/download/v0.01-alpha/TestModel.pth'


def TestModel(cache_dir=None):
    """
    Test model that runs very fast even on CPUs

    Performance is very bad!

    2 Conv layers

    5 RNN Layers with 400 hidden units


    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Pretrained DeepSpeech (Testing purposes) model
    :rtype: ``danspeech.deepspeech.model.DeepSpeech``
    """

    model_path = get_model(model_name="TestModel.pth", origin=MODEL_PACKAGE,
                           file_hash="c21438a33f847a9c8d4e08779e98bf31", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
