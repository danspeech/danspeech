from danspeech.deepspeech.model import DeepSpeech


def CustomModel(model_path):
    """
    Instantiates customly trained models

    :param model_path: Path to custom trained model
    :return: Custom DeepSpeech model
    """
    model = DeepSpeech.load_model(model_path)
    return model

