from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


# ToDO: This link works but should be changed as the models are released.
MODEL_PACKAGE = 'https://github.com/Rasmusafj/models_development/raw/master/400units.pth'


def TestModel(cache_dir=None):
    """
    Test model that runs very fast even on CPUs

    Performance is however very bad!

    2 Conv layers
    5 RNN Layers with 400 hidden units

    :return: Pretrained DeepSpeech (Testing purposes) model
    """
    model_path = get_model(model_name="TestModel.pth", origin=MODEL_PACKAGE,
                           file_hash="c21438a33f847a9c8d4e08779e98bf31", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
