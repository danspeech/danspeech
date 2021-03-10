from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from danspeech.utils.data_utils import get_model

MODEL_PACKAGE = 'Not available yet...'


def getWav2Vec2CTCModel(cache_dir=None) -> Wav2Vec2ForCTC:
    """
    Wav2Vec2 CTC trained model

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Pretrained Wav2vec2 CTC  model
    :rtype: ``transformers.Wav2Vec2ForCTC``
    """
    model_path = get_model(model_name="wav2vec2CTC", origin=MODEL_PACKAGE, cache_dir=cache_dir,
                           ignore_validation_and_download=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    return model


def getWav2Vec2CTCTokenizer(cache_dir=None) -> Wav2Vec2CTCTokenizer:
    """
    Wav2Vec2CTCTokenizer from transformers. May be used to decode output of model or encode input.

    :param str cache_dir: If you wish to use custom directory to stash/cache your models. This is generally not
        recommended, and if left out, the DanSpeech models will be stored in the ``~/.danspeech/models/`` folder.

    :return: Wav2Vec2CTCTokenizer from transformers
    :rtype: ``transformers.Wav2Vec2CTCTokenizer``
    """
    model_path = get_model(model_name="wav2vec2CTC", origin=MODEL_PACKAGE, cache_dir=cache_dir,
                           ignore_validation_and_download=True)
    model = Wav2Vec2CTCTokenizer.from_pretrained(model_path, word_delimiter_token=" ")
    return model
