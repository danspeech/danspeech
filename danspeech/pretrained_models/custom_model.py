from danspeech.deepspeech.model import DeepSpeech


def CustomModel(model_path):
    """
    Instantiates customly trained models

    :param str model_path: Path to custom trained model
    :return: Custom DeepSpeech model.
    :rtype: ``danspeech.deepspeech.model.DeepSpeech``
    """
    model = DeepSpeech.load_model(model_path)
    return model

