from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model


# ToDo: Add model package link for release
MODEL_PACKAGE = "toDo"


def GPUStreamingRNN(cache_dir=None):
    """
    DanSpeech model with lookahead, which works as a streaming model.

    This model will not be able to follow a stream of data on regular CPUS. Hence, use a GPU

    2 conv layers
    5 RNN layers (not bidirectional) with 2000 units each

    context is 20

    :return: Pretrained DeepSpeech (Streaming for GPU) model
    """
    model_path = get_model(model_name="GPUStreamingRNN.pth", origin=MODEL_PACKAGE,
                           file_hash="8194f47f5c63c14c3587d42aa37d622d", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model
