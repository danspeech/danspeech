from danspeech.deepspeech.train_refactoring import train_model
from danspeech.deepspeech.model import DeepSpeech

if __name__ == '__main__':
    model = DeepSpeech()
    train_model(model, '/scratch/s134843/preprocessed_train/', 'scratch/s134843/preprocessed_validation')
