from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


# ToDo: Add model package link for release
MODEL_PACKAGE = "toDo"


def CPUStreamingRNN(cache_dir=None):
    """
    DanSpeech model with lookahead, which works as a real-time streaming model.

    This model runs on most modern CPUs.

    2 conv layers

    5 RNN layers (not bidirectional) with 800 units each

    Lookahead context is 20

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Pretrained DeepSpeech (Streaming for CPU) model
    :rtype: ``danspeech.deepspeech.model.DeepSpeech``

    """
    model_path = get_model(model_name="CPUStreamingRNN.pth", origin=MODEL_PACKAGE,
                           file_hash="ba514ec96b511c0797dc643190a80269", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
