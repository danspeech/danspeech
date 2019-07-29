import torch
from danspeech.deepspeech.model import DeepSpeech
from distiller import sparsity

if __name__ == '__main__':
    package = torch.load('/home/mcn/newModels/baidu_pruned_baseline.pth', map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model_package(package)

    print(model)
    print(model.audio_conf)

    for name, layer in model.named_parameters():
        print(name, layer.size(), sparsity(layer))


