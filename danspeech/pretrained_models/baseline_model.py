from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = "toDo"

# ToDO: Create
def Baseline(cache_dir=None):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="Baseline.pth", origin=MODEL_PACKAGE, file_hash="81f03fa8b922445494a56bcd774c69a7", cache_dir=cache_dir)
    model = DeepSpeech.load_model(model_path)
    return model

