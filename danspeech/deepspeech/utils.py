import os

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def to_np(x):
    return x.data.cpu().numpy()


class TensorBoardLogger(object):

    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)
        self.log_params = log_params

    def update(self, epoch, values):
        values = {
            'Avg. Train Loss': values["loss_results"][epoch],
            'WER': values["wer"][epoch],
            'CER': values["cer"][epoch]
        }

        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)

    def load_previous_values(self, start_epoch, values):
        for i in range(start_epoch):
            values = {
                'Avg. Train Loss': values["loss_results"][i],
                'WER': values["wer"][i],
                'CER': values["cer"][i]
            }

            self.tensorboard_writer.add_scalars(self.id, values, i + 1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt


def get_default_audio_config():
    return {
        "normalize": True,
        "sampling_rate": 16000,
        "window": "hamming",
        "window_stride": 0.01,
        "window_size": 0.02
    }
