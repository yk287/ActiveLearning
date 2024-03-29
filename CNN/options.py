import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=1, help='total number of training episodes')
        self.parser.add_argument('--show_every', type=int, nargs='?', default=500, help='How often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=250, help='How often to print scores')
        self.parser.add_argument('--print_model', type=bool, nargs='?', default=True, help='Prints the model being used')
        self.parser.add_argument('--batch', type=int, nargs='?', default=32, help='batch size to be used')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0001, help='learning rate')
        self.parser.add_argument('--beta1', type=int, nargs='?', default=0.9, help='beta1 for ADAM')
        self.parser.add_argument('--beta2', type=int, nargs='?', default=0.999, help='beta2 for ADAM')

        self.parser.add_argument('--lrelu_val', type=int, nargs='?', default=0.01, help='leaky Relu Value')
        self.parser.add_argument('--directory', type=str, nargs='?', default='img', help='directory where image gets saved')
        self.parser.add_argument('--model_path', type=str, nargs='?', default='/home/youngwook/.ray/models/mnist_cnn.pt', help='directory where inception model gets saved')

        self.parser.add_argument('--seed', type=int, nargs='?', default=1234, help='seed for reproducibility')

        # Active Learning Options
        self.parser.add_argument('--dropout_iters', type=int, nargs='?', default=10, help='Number of dropout iterations')
        self.parser.add_argument('--initial_n', type=int, nargs='?', default=100, help='number of labelled data to start with')
        self.parser.add_argument('--total_data', type=int, nargs='?', default=15000, help='number of labelled data to start with')
        self.parser.add_argument('--total_training_step', type=int, nargs='?', default=5,
                                 help='total number of active learning steps to take')
        self.parser.add_argument('--n_addition', type=int, nargs='?', default=50, help='number of labelled data to start with')
        self.parser.add_argument('--random_select', type=bool, nargs='?', default=False,
                                 help='whether to randomly select next batch')
        self.parser.add_argument('--AL_methods', type=str, nargs='?', default='LC',
                                 help='which methods to use. read README to see supported methods')
        #Discriminator Options
        self.parser.add_argument('--D_hidden', type=int, nargs='+', default=[256, 256], help='hidden layer configuration in a list form for D')
        self.parser.add_argument('--D_activation', type=str, nargs='?', default='lrelu', help='Activation function for the discriminator')
        self.parser.add_argument('--D_input_size', type=int, nargs='?', default=784, help='size of input for the discriminator')
        self.parser.add_argument('--D_output_size', type=int, nargs='?', default=1, help='size of output for the discriminator')

        self.parser.add_argument('--D_input_channel', type=int, nargs='?', default=1,
                                 help='number of input channels')
        self.parser.add_argument('--D_channel_up', type=int, nargs='?', default=32,
                                 help='how many channels to go up to initially in CNN')
        self.parser.add_argument('--num_classes', type=int, nargs='?', default=10, help='number of prediction class')
        self.parser.add_argument('--n_discrim_down', type=int, nargs='?', default=3, help='hidden layer configuration in a list form for D')
        self.parser.add_argument('--dropout', type=float, nargs='?', default=0.25, help='dropout rate')

        #Generator Options
        self.parser.add_argument('--G_hidden', type=int, nargs='+', default=[1024, 1024], help='hidden layer configuration in a list form for G')
        self.parser.add_argument('--G_activation', type=str, nargs='?', default='relu', help='Activation function for the generator')
        self.parser.add_argument('--noise_dim', type=int, nargs='?', default=96, help='size of noise input for the generator')
        self.parser.add_argument('--G_output_size', type=int, nargs='?', default=784, help='size of output for the discriminator')
        self.parser.add_argument('--G_out_activation', type=str, nargs='?', default='tanh', help='final output activator')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt