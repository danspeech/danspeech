from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


# ToDo: Add model package link for release
MODEL_PACKAGE = 'toDO'


def Folketinget(cache_dir=None):
    """
    The deepest and best performing DanSpeech model adapted to data from Folketinget.

    3 Conv layers
    9 RNN Layers with 1200 hidden units

    WARNING: This model is really slow, so we suggest you use it with a GPU.

    :return: Pretrained DeepSpeech (Folketinget tuned) model
    """
    model_path = get_model(model_name="Folketinget.pth", origin=MODEL_PACKAGE,
                           file_hash="9523d5744ad4ff5ffc8519393350cc91", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
