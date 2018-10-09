import torch
import re
import numpy as np
import argparse

from scipy import io as sio
from tqdm import tqdm

# code adapted from https://github.com/bilylee/SiamFC-TensorFlow/blob/master/utils/train_utils.py
def convert(mat_path):
    """Get parameter from .mat file into parms(dict)"""

    def squeeze(vars_):
    # Matlab save some params with shape (*, 1)
    # However, we don't need the trailing dimension in TensorFlow.
        if isinstance(vars_, (list, tuple)):
            return [np.squeeze(v, 1) for v in vars_]
        else:
            return np.squeeze(vars_, 1)

    netparams = sio.loadmat(mat_path)["net"]["params"][0][0]
    params = dict()

    name_map = {(1, 'conv'): 0,  (1, 'bn'): 1,
                (2, 'conv'): 4,  (2, 'bn'): 5,
                (3, 'conv'): 8,  (3, 'bn'): 9,
                (4, 'conv'): 11, (4, 'bn'): 12,
                (5, 'conv'): 14}
    for i in tqdm(range(netparams.size)):
        param = netparams[0][i]
        name = param["name"][0]
        value = param["value"]
        value_size = param["value"].shape[0]

        match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name, re.I)
        if match:
            items = match.groups()
        elif name == 'adjust_f':
            continue
        elif name == 'adjust_b':
            params['corr_bias'] = torch.from_numpy(squeeze(value))
            continue


        op, layer, types = items
        layer = int(layer)
        if layer in [1, 2, 3, 4, 5]:
            idx = name_map[(layer, op)]
            if op == 'conv':  # convolution
                if types == 'f':
                    params['features.{}.weight'.format(idx)] = torch.from_numpy(value.transpose(3, 2, 0, 1))
                elif types == 'b':#  and layer == 5:
                    value = squeeze(value)
                    params['features.{}.bias'.format(idx)] = torch.from_numpy(value)
            elif op == 'bn':  # batch normalization
                if types == 'x':
                    m, v = squeeze(np.split(value, 2, 1))
                    params['features.{}.running_mean'.format(idx)] = torch.from_numpy(m)
                    params['features.{}.running_var'.format(idx)] = torch.from_numpy(np.square(v))
                    # params['features.{}.num_batches_tracked'.format(idx)] = torch.zeros(0)
                elif types == 'm':
                    value = squeeze(value)
                    params['features.{}.weight'.format(idx)] = torch.from_numpy(value)
                elif types == 'b':
                    value = squeeze(value)
                    params['features.{}.bias'.format(idx)] = torch.from_numpy(value)
            else:
                raise Exception
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_path', type=str, default="./models/2016-08-17.net.mat")
    args = parser.parse_args()
    params = convert(args.mat_path)
    torch.save(params, "./models/siamfc_pretrained.pth")
