

import numpy as np

import torch

# get device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import random

from models import discriminator

from activeLearner import activeLearner

from utils import plot_lines

def main(opts):

    # make everything deterministic
    torch.manual_seed(opts.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(opts.seed)
    np.random.seed(opts.seed)

    al = activeLearner(opts)
    al.AL_train()

    return al.train_accuracy, al.valid_accuracy, al.data_size

if __name__ == '__main__':
    # options
    from options import options

    options = options()
    opts = options.parse()

    # run through all the methods

    al_methods = ['RANDOM', 'LC', 'DROPOUT_LC', 'MAX_ENTROPY', 'DROPOUT_MAX_ENTROPY']

    valid_metrics = []
    train_metrics = []

    for method in al_methods:

        print("Working on {}".format(method))

        opts.AL_methods = method
        train_metric, valid_metric, data_size = main(opts)

        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)

    plot_lines(data_size, valid_metrics, al_methods, title='Validation Accuracy')
    plot_lines(data_size, train_metrics, al_methods, title='Training Accuracy')




