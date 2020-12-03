import json
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from danspeech.errors.model_errors import ConvError, FreezingMoreLayersThanExist
from danspeech.deepspeech.utils import get_default_audio_config


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    """
    Source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    """

    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    """
    Source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    """

    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    """
    Source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

    """
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    """
    Source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    """

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    """
    Source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

    Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    input shape - sequence, batch, feature - TxNxH
    output shape - same as input
    """

    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class MaskConvStream(nn.Module):
    def __init__(self, seq_module):
        """
        We are not masking anymore, since it should only handle single sequence.

        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConvStream, self).__init__()
        self.seq_module = seq_module
        self.left_1 = None
        self.left_2 = None

    def forward(self, x, is_first, is_last):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """

        # First requires zero padding on left size for both conv layers
        for i, module in enumerate(self.seq_module):

            # Paddings for first and last
            if is_first and (i == 0 or i == 3):
                x = F.pad(x, pad=(5, 0), value=0)  # Zero padding left
            elif is_last and (i == 0 or i == 3):
                x = F.pad(x, pad=(0, 5), value=0)  # Zero padding ending

            if not is_first:
                if i == 0:
                    # print(self.left_1[:,:,:,:])
                    x = torch.cat([self.left_1, x], dim=3)
                    # print(x[:,:,:,0:10])
                if i == 3:
                    x = torch.cat([self.left_2, x], dim=3)
            # Store for next chunk (precomputed)
            if not is_last:
                if i == 0:
                    self.left_1 = x[:, :, :, -10:]  # Store for next chunk
                elif i == 3:
                    self.left_2 = x[:, :, :, -10:]  # Store for next chunk

            x = module(x)

        return x


class BatchRNNStream(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, batch_norm=True):
        super(BatchRNNStream, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=False, bias=True)
        self.num_directions = 1
        self.previous_hidden = None
        self.previous_init = False

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, is_last):

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if not self.previous_init:
            x, h = self.rnn(x)
            self.previous_init = True
        else:
            x, h = self.rnn(x, hx=self.previous_hidden)
        # x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x)

        self.previous_hidden = h
        if is_last:
            self.previous_hidden = None
            self.previous_init = False

        return x


class LookaheadStream(nn.Module):
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(LookaheadStream, self).__init__()
        self.n_features = n_features
        self.context = context
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

        self.hidden_initiated = False
        self.hidden_states_buffer = None

        self.hard_tanh = nn.Hardtanh(0, 20, inplace=True)

    def forward(self, x, is_last, is_first):
        if not self.hidden_initiated or is_first:
            self.hidden_states_buffer = x
            self.hidden_initiated = True
            return torch.tensor(0)  # Dummy return
        else:
            out = torch.cat([self.hidden_states_buffer, x], dim=0)
            # Save the last context-1 they cannot be computed until next pass
            self.hidden_states_buffer = x[-(self.context-1):, :, :]

        out = out.transpose(0, 1).transpose(1, 2)

        if is_last:
            out = F.pad(out, pad=self.pad, value=0)

        out = self.conv(out)
        out = out.transpose(1, 2).transpose(0, 1).contiguous()

        out = self.hard_tanh(out)

        if is_last:
            self.hidden_initiated = False
            self.hidden_states_buffer = None

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    """
    Source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Modified for DanSpeech
    """

    def __init__(self, model_name, rnn_type=nn.GRU, labels=None, rnn_hidden_size=768, rnn_layers=5, audio_conf=None,
                 bidirectional=True, context=20, conv_layers=2, streaming_inference_model=False):

        """
        Init function of a DeepSpeech model for DanSpeech.

        :param model_name: String, String representing the model name.
        :param rnn_type: torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN
        :param labels: String, Label set to be used with the model. If none given, will default to DanSpeech labels.
        :param rnn_hidden_size: Int, Number of hidden units size of all recurrent layers.
        :param rnn_hidden_layers: Int, Number of reccurent layers.
        :param audio_conf: Dict, Audio configuraton for the input Spectrograms. If none given, will default to the
        DanSpeech settings.
        :param bidirectional: Boolean, Whether to use bidirectional or not. For performance, we recommend using
        bidirectional but if you need a streaming model, then give False. The model will then use a LookAhead layer
        On top of all the RNN layers.
        :param context: Int, Context of the LookAhead layer i.e. how much do we look forward. Ignored if Bidirectional
        is True.
        :param conv_layers: Int, number of convolutional layers. 1,2,3 is currently only supported.
        :param streaming_inference_model: Boolean, indication if the model is a streaming inference model. Default is
        False.
        """
        super(DeepSpeech, self).__init__()

        # Init labels if they are not given.
        if not labels:
            from . import __path__ as ROOT_PATH
            label_path = os.path.join(ROOT_PATH[0], "labels.json")
            with open(label_path, "r", encoding="utf-8") as label_file:
                labels = str(''.join(json.load(label_file)))

        # Init default audio config if not given
        if audio_conf is None:
            audio_conf = get_default_audio_config()

        # model metadata needed for serialization/deserialization
        self.model_name = model_name
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.audio_conf = audio_conf or {}
        self.labels = labels
        self.bidirectional = bidirectional
        self.conv_layers = conv_layers
        self.streaming_model = streaming_inference_model
        self.context = context

        sample_rate = self.audio_conf.get("sampling_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)
        num_classes = len(self.labels)

        if conv_layers == 0:
            raise ConvError("0 convolutional layers configuration not supported by DanSpeech")

        if conv_layers > 3:
            raise ConvError("Maximum amount of convolutional layers supported by DanSpeech is 3")

        if self.streaming_model:
            self.streaming_init(conv_layers, rnn_hidden_size, rnn_type, rnn_layers, context)
        else:
            # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
            rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)

            if conv_layers == 1:
                self.conv = MaskConv(nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                )

                rnn_input_size *= 32

            elif conv_layers == 2:
                self.conv = MaskConv(nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True)
                )
                )

                rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
                rnn_input_size *= 32

            elif conv_layers == 3:
                self.conv = MaskConv(nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                    nn.BatchNorm2d(96),
                    nn.Hardtanh(0, 20, inplace=True)
                )
                )
                rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
                rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
                rnn_input_size *= 96

            rnns = []
            rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, batch_norm=False)
            rnns.append(('0', rnn))
            for x in range(rnn_layers - 1):
                rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                               bidirectional=bidirectional)
                rnns.append(('%d' % (x + 1), rnn))
            self.rnns = nn.Sequential(OrderedDict(rnns))
            self.lookahead = nn.Sequential(
                # consider adding batch norm?
                Lookahead(rnn_hidden_size, context=context),
                nn.Hardtanh(0, 20, inplace=True)
            ) if not bidirectional else None

        # Always use this configuration
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

        # Forward should now be streaming forward
        if self.streaming_model:
            self.forward = self.streaming_forward

    def streaming_init(self, conv_layers, rnn_hidden_size, rnn_type, nb_layers, context):

        sample_rate = self.audio_conf.get("sampling_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)

        if conv_layers == 1:
            self.conv = MaskConv(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
            )
            )

            rnn_input_size *= 32

        elif conv_layers == 2:
            self.conv = MaskConvStream(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )
            )

            rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
            rnn_input_size *= 32

        elif conv_layers == 3:
            self.conv = MaskConvStream(nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(96),
                nn.Hardtanh(0, 20, inplace=True)
            )
            )
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
            rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
            rnn_input_size *= 96

        sample_rate = self.audio_conf.get("sample_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNNStream(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                             batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchRNNStream(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = LookaheadStream(rnn_hidden_size, context=context)

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def streaming_forward(self, x, is_first, is_last):

        x = self.conv(x, is_first, is_last)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, is_last)

        x = self.lookahead(x, is_last, is_first)
        # If x returned is none, then the layer is buffering for processing
        if len(x.size()) < 2:
            return None

        x = self.fc(x)
        x = x.transpose(0, 1)

        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x


    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()

    def freeze_layers(self, number_to_freeze=0):
        """
        Freezes x amounts of layers in the model for training i.e. weights will not be updated.
        :param number_to_freeze: int
        """
        counter = 0

        # If no layers to freeze, return
        if number_to_freeze == 0:
            return

        # If trying to freeze more layers than there is, throw error
        if number_to_freeze > self.conv_layers + self.rnn_layers:
            raise FreezingMoreLayersThanExist("You are trying to freeze more layers than exists in "
                                              "model... Choose smaller number")

        for name, child in self.named_children():
            # Conv layers consist of 3 blocks
            if name == "conv":
                if counter < number_to_freeze:
                    seq_child = child.named_children()
                    name, seq_module = next(seq_child)
                    conv_module_childs = list(seq_module.named_children())
                    # Since conv layer consist of 3 blocks
                    number_of_conv_layers = int(len(conv_module_childs) / 3)
                    # Either freeze all conv layers (first min) or specified number if less than actual amount (second min)
                    number_of_conv_children_to_freeze = min(number_of_conv_layers * 3, number_to_freeze * 3)
                    # Freeze all 3 blocks.
                    for i in range(number_of_conv_children_to_freeze):
                        if i % 3 == 0:
                            print("Freezing conv layer {}".format(counter))
                            counter += 1

                        name, conv_child = conv_module_childs[i]
                        for param in conv_child.parameters():
                            param.requires_grad = False

            if name == "rnns":
                rnn_childs = child.named_children()
                for name, rnn_child in rnn_childs:
                    if counter < number_to_freeze:
                        print("Freezing rnn layer {}".format(counter))
                        for param in rnn_child.parameters():
                            param.requires_grad = False
                    counter += 1

    @classmethod
    def load_model(cls, path):
        """
        If you do not need meta data from package, simply use this method to load the model.

        :param path: Path to .pth package for the model.
        :return: DeepSpeech DanSpeech trained model
        """
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(model_name=package['model_name'],
                    rnn_hidden_size=package['rnn_hidden_size'],
                    rnn_layers=package['rnn_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package['bidirectional'],
                    conv_layers=package['conv_layers'],
                    context=package['context'],
                    streaming_inference_model=package["streaming_model"])

        model.load_state_dict(package['state_dict'])

        # This is important to make sure that the weights are initialized correctly into the memory
        for x in model.rnns:
            x.flatten_parameters()
        return model

    @classmethod
    def load_model_package(cls, package):
        """
        If you need aditional information from package, then use this loading method instead.

        :param package: Package holding the model,
        :return: DeepSpeech DanSpeech trained model
        """
        model = cls(model_name=package['model_name'],
                    rnn_hidden_size=package['rnn_hidden_size'],
                    rnn_layers=package['rnn_layers'],
                    labels=package['labels'],
                    audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']],
                    bidirectional=package['bidirectional'],
                    conv_layers=package['conv_layers'],
                    context=package['context'],
                    streaming_inference_model=package["streaming_model"])

        model.load_state_dict(package['state_dict'])

        # This is important to make sure that the weights are initialized correctly into the memory
        for x in model.rnns:
            x.flatten_parameters()
        return model

    @staticmethod
    def get_param_size(model):
        """
        Get the number of parameters for a given model.

        :param model: DeepSpeech model
        :return: Number of params
        """
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
