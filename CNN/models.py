import torch
import torch.nn as nn
import torch.nn.functional as F

class discriminator(nn.Module):
    '''
    Patch Discriminator used in GANS.
    '''

    def __init__(self, opts):
        super(discriminator, self).__init__()

        self.opts = opts
        steps = []
        in_channel = opts.D_input_channel
        channel_up = opts.D_channel_up

        for i in range(opts.n_discrim_down):
            steps += [nn.Conv2d(in_channel, channel_up, 4, 2, 1), nn.BatchNorm2d(channel_up), nn.LeakyReLU(opts.lrelu_val, True), nn.Dropout2d(opts.dropout)]
            in_channel = channel_up
            channel_up *= 2

        steps += [nn.Conv2d(in_channel, opts.num_classes, 5, 1, 1)]

        steps += [nn.Softmax()]

        self.model = nn.Sequential(*steps)

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """

        self.model.eval() # turn off other things

        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() # turn on dropouts


    def n_dropout(self, x):

        # turn only dropouts on
        self.enable_dropout()

        # empty placeholder for results
        results = torch.zeros((self.opts.batch, self.opts.num_classes))

        # iterate through and get results
        for _ in range(self.opts.dropout_iters):
            results += self.forward(x)

        return results

    def n_dropout_least_confident(self, x):

        results = self.n_dropout(x)

        # average the probabilities
        avg_prob = torch.divide(results, torch.tensor(self.opts.dropout_iters))


    def n_dropout_max_entropy(self, x):

        results = self.n_dropout(x)

        # average the probabilities
        avg_prob = torch.divide(results, torch.tensor(self.opts.dropout_iters))

        # get the log2 of prob
        log_2_avg_prob = torch.log2(avg_prob)
        # get the entropy
        entropy_avg_pi = - torch.multiply(avg_prob, log_2_avg_prob)
        # sum to get the entropy
        entropy_pi = torch.sum(entropy_avg_pi, dim=1)
        # change back to train mode()
        self.model.train()

        return entropy_pi


    def forward(self, x):

        return self.model(x).reshape(-1, self.opts.num_classes)