

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import discriminator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load mnist dataset and define network
from torchvision import datasets, transforms

import numpy as np

from sklearn.model_selection import train_test_split


class activeLearner():

    def __init__(self, opts):

        self.opts = opts

        self.pick_initial_n()
        self.init_dataloader()

    def pick_initial_n(self):

        # Define a transform to normalize the data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.transforms.Normalize((0.5), (0.5))
        ])

        # Download and load the training data
        self.dataset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)

        # Split the indices in a stratified way
        indices = np.arange(len(self.dataset))
        self.train_indices, self.left_over_indices = train_test_split(indices, train_size=self.opts.initial_n, stratify=self.dataset.targets, random_state= self.opts.seed)

        # limit
        self.left_over_indices = self.left_over_indices[:self.opts.total_data]

        # Warp into Subsets and DataLoaders
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        self.left_over = torch.utils.data.Subset(self.dataset, self.left_over_indices)

    def init_dataloader(self):
        '''
        A function that (re)-initializes dataloader given the dataset
        :return:
        '''

        self.trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.opts.batch, shuffle=True, drop_last=True)
        self.validloader = torch.utils.data.DataLoader(self.left_over, batch_size=self.opts.batch, shuffle=True, drop_last=True)

    def update_indices(self):

        # find candidates that need to be labeled
        self.find_candidates()

        # update the dataloader
        self.train_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)
        self.left_over = torch.utils.data.Subset(self.dataset, self.left_over_indices)

    def AL_train(self):

        for i in range(10):
            # initial train
            self.train_model()
            # update the train and test indices by updating indices
            self.update_indices()
            # reload the dataloader
            self.init_dataloader()

    def find_candidates(self):

        if not self.opts.random_select:
            # placeholder
            results = torch.zeros((0))

            for image, label in self.validloader:

                # get the criteria for acquisition function.
                result = self.model.n_dropout_max_entropy(image.to(device)).detach()

                # append the data
                results = torch.concat((results, result.cpu()), dim = 0)

            # get a new set of candidates
            self.candidate_indices = torch.argsort(results, descending=True)#[:self.opts.n_addition].tolist()
            self.candidate_indices = self.candidate_indices.tolist()[:self.opts.n_addition]

        else:

            perm = torch.randperm(len(self.left_over_indices))
            self.candidate_indices = perm[:self.opts.n_addition]

            #self.candidate_indices = self.left_over_indices[idx]


        # candidate_indices represent the data that should be added to the train_dataset
        self.candidate_indices = self.left_over_indices[self.candidate_indices]

        # update the train_indices
        self.train_indices = np.concatenate((self.train_indices, np.asarray(self.candidate_indices)), 0)

        # update the left_over_indices.
        self.left_over_indices = [item for item in self.left_over_indices if item not in self.candidate_indices]

        self.left_over_indices = np.asarray(self.left_over_indices)

        # make sure there are no overlapping indices between training and testing
        assert len(np.intersect1d(np.asarray(self.left_over_indices), np.asarray(self.train_indices))) == 0

    def train_model(self):
        '''
        A function that trains the model given the (updated) dataset
        :return:
        '''

        self.model = discriminator(self.opts)

        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.opts.lr, betas=(self.opts.beta1, self.opts.beta2))
        criterion = torch.nn.CrossEntropyLoss()

        # train the model
        for i in range(1):#self.opts.epoch):

            self.model.train()

            for image, label in self.trainloader:
                # zero out the gradient
                optimizer.zero_grad()

                # pred
                probs = self.model(image.to(device))

                loss = criterion(probs, label.to(device))
                loss.backward()
                optimizer.step()

            # Validation

            correct = 0
            total = 0
            self.model.eval()
            for image, label in self.validloader:


                # pred
                probs = self.model(image.to(device))

                pred = torch.argmax(probs, dim=1)

                correct += torch.sum((pred == label.to(device)) * 1.0)
                total += len(pred)

            print("Valid Accuracy: ", float(correct / total))

