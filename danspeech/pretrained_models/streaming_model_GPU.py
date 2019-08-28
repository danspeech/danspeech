from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "https://github.com/danspeech/danspeech/releases/download/v0.01-alpha/GPUStreamingRNN.pth"


def GPUStreamingRNN(cache_dir=None):
    """
    DanSpeech model with lookahead, which works as a real-time streaming model.

    This model will not be able to follow a stream of data on regular CPUS. Hence, use a GPU.

    2 conv layers

    5 RNN layers (not bidirectional) with 2000 units each

    Lookahead context is 20

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Pretrained DeepSpeech (Streaming for GPU) model
    :rtype: ``danspeech.deepspeech.model.DeepSpeech``
    """
    model_path = get_model(model_name="GPUStreamingRNN.pth", origin=MODEL_PACKAGE,
                           file_hash="8194f47f5c63c14c3587d42aa37d622d", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
