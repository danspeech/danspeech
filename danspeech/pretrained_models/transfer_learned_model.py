from danspeech.deepspeech.model import DeepSpeech
from danspeech.utils.data_utils import get_model

# ToDO: Repackage and include model

MODEL_PACKAGE = 'https://github.com/Rasmusafj/models_development/raw/master/400units.pth'

def TransferLearned():
    """
    Instantiates model with 2 conv layers and 5 rnn layers each with 400 units

    :return: Pretrained DeepSpeech model
    """
    model_path = get_model(model_name="TransferLearned.pth", origin=MODEL_PACKAGE,
                           file_hash="d19b9d7dc976bffbc9225e0f80ecacbf")
    model = DeepSpeech.load_model(model_path)
    return model
