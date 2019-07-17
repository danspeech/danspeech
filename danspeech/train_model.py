import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
sys.argv.pop(1)

import argparse
import random
import numpy as np
import torch

from danspeech.deepspeech.train import train_model

print(os.environ["CUDA_VISIBLE_DEVICES"])

if torch.cuda.is_available():
    print("Success")
    print(torch.cuda.device_count())

# -- DanSpeech-related arguments
parser = argparse.ArgumentParser(description='DanSpeech Training / Pruning')

# -- Data-specific arguments, required!
parser.add_argument('--train-data-path', metavar='DIR',
                    help='Path to training data', default=None)
parser.add_argument('--validation-data-path', metavar='DIR',
                    help='Path to validation data', default=None)
parser.add_argument('--labels-path', default='./labels.json',
                    help='Contains all characters for transcription')

# -- Spectogram hyper-parameters, defaults to DanSpeech model parameters
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')

# -- Network-specific hyper-parameters, defaults to DanSpeech baseline model parameters
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--learning-anneal', default=1.0, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')

# -- Model-monitoring related arguments
parser.add_argument('--id', default='DanSpeech_dummy_model',
                    help='File name given to checkpoint models')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing',
                    default=True)

# -- GPU-training arguments
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--num-workers', default=6, type=int, help='Number of workers used in data-loading')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=1337, type=int, help='Seed to generators')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')

# -- Training specific arguments
parser.add_argument('--continue-from', default='', help='Continue training from checkpoint')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint by freezing all layers except the output layer')

# -- Multiple-GPU training arguments
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')

# -- Distiller-related pruning arguments
parser.add_argument('--prune', dest='prune', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (dafault is to use hard-coded schedule)')

if __name__ == '__main__':

    args = parser.parse_args()
    args.distributed = args.world_size > 1
    args.cuda = True

    from danspeech.deepspeech.model import DeepSpeech
    model = DeepSpeech()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Begin training
    train_model(model, args)
