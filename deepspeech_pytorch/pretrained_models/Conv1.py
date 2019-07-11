
from deepspeech_pytorch.model import DeepSpeech

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5')

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


# ToDO: Top weights option
def Conv1():
    """
    Instantiates the most complex DanSpeech model with a lot of parameters

    :return: Pretrained DeepSpeech model
    """

    model = DeepSpeech(conv_layers=3,
                       rnn_hidden_size=1200,
                       nb_layers=9
                       )

    DeepSpeech.load_model()