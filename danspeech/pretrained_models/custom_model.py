
from danspeech.deepspeech.model import DeepSpeech

# ToDO: Create
def Baseline(model_path):
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """
    model = DeepSpeech.load_model(model_path)
    return model

