import os

import hashlib
import wget


def _hash_file(fpath, chunk_size=65535):
    """
    Calculates the md5 hash of a file

    :param fpath: path to the file being validated
    :param chunk_size: Bytes to read at a time (leave unless large files)
    :return: The file hash
    """
    hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, chunk_size=65535):
    """
    Validates a file against a md5 hash.

    :param fpath: path to the file being validated
    :param file_hash: The expected md5 hash string of the file.
    :param chunk_size: Bytes to read at a time (leave unless large files)
    :return: Whether the file is valid
    """
    if str(_hash_file(fpath, chunk_size)) == str(file_hash):
        return True
    else:
        return False


subdir_mapper = {"acoustic_model": "models",
                 "language_model": "lms"}


def get_model(model_name,
              origin,
              file_type="acoustic_model",
              file_hash=None,
              cache_dir=None):
    """
    Downloads a model file from a URL if it not already in the cache.

    :param file_type: acoustic_model or language_model. Determines subdir to store model.
    :param model_name: Name of the model
    :param origin: URL to the model package
    :param file_hash: The expected md5 hash string of the file. Use None if no validation is needed,
    :param cache_dir: Where to save models. Defaults to ~/.danspeech/models/
    :return: The local filepath to the model
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.danspeech', subdir_mapper[file_type])

    hash_algorithm = 'md5'
    os.makedirs(cache_dir, exist_ok=True)

    download = False
    fpath = os.path.join(cache_dir, model_name)

    if os.path.exists(fpath) and file_hash:
        if not validate_file(fpath, file_hash):
            print('A local file was found, but it seems to be '
                  'incomplete or outdated because the ' + hash_algorithm +
                  'file hash does not match the original value of ' +
                  file_hash + 'hence the model will be redownloaded and '
                              'the incomplete or outdated model will be deleted')
            download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        try:
            wget.download(url=origin, out=fpath)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise e

    return fpath
