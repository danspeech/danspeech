from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = 'https://github.com/Rasmusafj/models_development/raw/master/400units.pth'

def TestModel():
    """
    Instantiates model with 2 conv layers and 5 rnn layers each with 400 units

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="400_units.pth", origin=MODEL_PACKAGE,
                           file_hash="1bb9c5b6bb8259193b5ac7e2e0490cd7")
    model = DeepSpeech.load_model(model_path)
    return model
