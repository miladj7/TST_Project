import numpy as np
import pickle
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch

dirname = os.path.dirname(__file__)


def blobs(theta, samplesize, spots=(3, 3), sigma=[0.1, 0.3]):
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    gaussians = np.array([np.random.normal(0, sigma[0], samplesize), np.random.normal(0, sigma[1], samplesize)])
    data = rotMatrix @ gaussians
    shifts = [np.random.randint(0, spots[0], samplesize), np.random.randint(0, spots[1], samplesize)]
    data = np.add(data, shifts)
    return np.transpose(data)


def generate_samples(dataset, hypothesis, samplesize,data_all=None,data_trans=None):
    assert dataset in ['diff_var', 'diff_mean', 'diff_cov', 'mnist', 'fake_mnist', 'blobs','cifar'], 'unknown dataset'
    if dataset == 'diff_var':
        if hypothesis == 'alternative':
            x = np.random.normal(loc=0., scale=1., size=samplesize)
            y = np.random.normal(loc=0., scale=1.25, size=samplesize)
        if hypothesis == 'null':
            x = np.random.normal(loc=0., scale=1., size=samplesize)
            y = np.random.normal(loc=0., scale=1., size=samplesize)
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
    if dataset == 'diff_mean':
        d = 100
        x = []
        y = []
        if hypothesis == 'alternative':
            for i in range(samplesize):
                P = np.append(np.random.normal(loc=1, size=1), np.random.normal(loc=0, size=d - 1))
                x.append(P)
                Q = list(np.random.normal(loc=0, size=d))
                y.append(Q)
            x = np.array(x)
            y = np.array(y)
        if hypothesis == 'null':
            for i in range(samplesize):
                P = list(np.random.normal(loc=0, size=d))
                x.append(P)
                Q = list(np.random.normal(loc=0, size=d))
                y.append(Q)
            x = np.array(x)
            y = np.array(y)

    if dataset == 'diff_cov':
        d = 100
        x = []
        y = []
        Sigma = np.eye(2)
        Sigma[0,1] = 0.64
        Sigma[1, 0] = 0.64
        if hypothesis == 'alternative':
            for i in range(samplesize):
                P = np.append(np.random.multivariate_normal(np.zeros(2), Sigma), np.random.normal(loc=0, size=d - 2))
                x.append(P)
                Q = list(np.random.normal(loc=0, size=d))
                y.append(Q)
            x = np.array(x)
            y = np.array(y)
        if hypothesis == 'null':
            for i in range(samplesize):
                P = list(np.random.normal(loc=0, size=d))
                x.append(P)
                Q = list(np.random.normal(loc=0, size=d))
                y.append(Q)
            x = np.array(x)
            y = np.array(y)


    if dataset == 'blobs':
        if hypothesis == 'alternative':
            x = blobs(theta=0, samplesize=samplesize)
            y = blobs(theta=1.57, samplesize=samplesize)
        if hypothesis == 'null':
            x = blobs(theta=0, samplesize=samplesize)
            y = blobs(theta=0, samplesize=samplesize)

    if dataset == 'mnist':
        # Note: this is not how we used it for our experiments. It reloads the whole dataset every time. If you want to
        # run a lot of experiments with the mnist dataset, consider loading the data once and then simply
        # drawing new random indices every iteration.

        # load data
        # you must run the file 'download_mnist.py' before. This creates the with the 7x7 images
        file = os.path.join(dirname, 'mnist_7x7.data')
        with open(file, 'rb') as handle:

            X = pickle.load(handle)
        # define the distributions
        if hypothesis == 'null':
            P = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['8'], X['9']))
            Q = np.copy(P)
        if hypothesis == 'alternative':
            P = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['8'], X['9']))
            Q = np.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))

        # sample randomly from the datasets
        idx_X = np.random.randint(len(P), size=samplesize)
        x = P[idx_X, :]
        idx_Y = np.random.randint(len(Q), size=samplesize)
        y = Q[idx_Y, :]

    if dataset == 'fmnist':
        # Prepare real MNIST
        dataloader_FULL = torch.utils.data.DataLoader(
            datasets.MNIST(
                dirname,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=60000,
            shuffle=True,
        )
        # Obtain real MNIST images
        for i, (imgs, Labels) in enumerate(dataloader_FULL):
            data_all = imgs

        idx_X = np.random.choice(len(data_all), samplesize, replace=False)
        x = data_all[idx_X]
        if hypothesis == 'null':
            idx_Y = np.random.choice(len(data_all), samplesize, replace=False)
            y = data_all[idx_Y]
        if hypothesis == 'alternative':
            Fake_MNIST = pickle.load(open(os.path.join(dirname, 'Fake_MNIST_data_EP100_N10000.pckl', 'rb')))
            ind_Y = np.random.choice(4000, samplesize, replace=False)
            y = torch.from_numpy(Fake_MNIST[0][ind_Y])

        x = x.reshape(samplesize, -1)
        y = y.reshape(samplesize, -1)

    if dataset == 'cifar':
        if hypothesis == 'null':
            Ind_0 = np.random.choice(len(data_all), samplesize, replace=False)
            x = data_all[Ind_0]
            Ind_1 = np.random.choice(len(data_all), samplesize, replace=False)
            y = data_all[Ind_1]
        if hypothesis == 'alternative':
            Ind_0 = np.random.choice(len(data_all), samplesize, replace=False)
            x = data_all[Ind_0]
            Ind_1 = np.random.choice(data_trans.shape[0], samplesize, replace=False)
            y = data_trans[Ind_1]

    if dataset == 'own_dataset':
        '''
        To utilize your owm distributions please put samples into the  numpy arrays x and y of shape n x p,
        where n is the (equal) samplesize  and p the dimensionality of your data'''
        # x = np.array( YOUR SAMPLES FROM P HERE )
        # y = np.array( YOUR SAMPLES FROM Q HERE )


    return x, y
