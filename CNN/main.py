

import numpy as np

import torch

# get device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import random

from models import discriminator

from activeLearner import activeLearner

def main(opts):

    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    np.random.seed(opts.seed)

    al = activeLearner(opts)
    al.train_model()

    al.find_candidates()

    print('hello')


if __name__ == '__main__':
    # options
    from options import options

    options = options()
    opts = options.parse()

    main(opts)

