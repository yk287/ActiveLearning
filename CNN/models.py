import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    '''
    DROPOUT START
    '''
    def n_dropout(self, x):

        # turn only dropouts on
        self.enable_dropout()

        # empty placeholder for results
        results = torch.zeros((self.opts.batch, self.opts.num_classes)).to(device)

        # iterate through and get results
        for _ in range(self.opts.dropout_iters):
            results += self.forward(x)

        # average the probabilities
        avg_prob = torch.divide(results, torch.tensor(self.opts.dropout_iters))

        return avg_prob

    def n_dropout_least_confident(self, x):

        results = self.n_dropout(x)

        # get the least confident probabilities
        least_confident = self.calc_least_confident(results)

        # change back to train mode()
        self.model.train()

        return least_confident

    def n_dropout_max_entropy(self, x):

        # get the results
        results = self.n_dropout(x)

        # get the max entropy
        entropy_pi = self.calc_max_entropy(results)

        self.model.train()

        return entropy_pi

    '''
    DROPOUT END
    '''

    '''
    LEAST CONFIDENT START
    '''

    def get_least_confident(self, x):

        self.model.eval()
        # forward pass
        result = self.forward(x)

        least_confident = self.calc_least_confident(result)

        self.model.train()

        return least_confident


    def calc_least_confident(self, probabilities):
        '''
        given a tensor of shape B X O, return B X 1 which represents the least confident probabilities ( 1 - argmax_x( P(Y|x))
        '''

        # get the least confident probabilities
        least_confident = 1.0 - probabilities.max(dim=1)[0]

        return least_confident
    '''
    LEAST CONFIDENT END
    '''

    '''
    MAX ENTROPY START
    '''
    def get_max_entropy(self, x):

        self.model.eval()
        # forward pass
        result = self.forward(x)

        self.model.train()

        return self.calc_max_entropy(result)

    def calc_max_entropy(self, probabilities):

        # get the log2 of prob
        log_2_avg_prob = torch.log2(probabilities)
        # get the entropy
        entropy_avg_pi = - torch.multiply(probabilities, log_2_avg_prob)
        # sum to get the entropy
        entropy_pi = torch.sum(entropy_avg_pi, dim=1)
        # change back to train mode()

        return entropy_pi

    '''
    MAX ENTROPY END
    '''


    def forward(self, x):

        return self.model(x).reshape(-1, self.opts.num_classes)