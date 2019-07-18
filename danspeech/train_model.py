from danspeech.deepspeech.train_refactoring import continue_training
from danspeech.pretrained_models import Units400

if __name__ == '__main__':
    model = Units400()
    print(model)
    print(model.audio_conf)
    continue_training(model, '/scratch/s134843/preprocessed_train/', '/scratch/s134843/preprocessed_validation/')
