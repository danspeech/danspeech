from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = 'https://github.com/Rasmusafj/models_development/raw/master/400units.pth'

# ToDO: Create
def Folketinget():
    """
    Instantiates model with 2 conv layers and 5 rnn layers each with 400 units

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="Folketinget.pth", origin=MODEL_PACKAGE,
                           file_hash="9523d5744ad4ff5ffc8519393350cc91")
    model = DeepSpeech.load_model(model_path)
    return model
