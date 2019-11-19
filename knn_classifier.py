import numpy as np
import torch
import unittest
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import cs236781.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None
        self.test = unittest.TestCase()

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        x_train_list = []
        y_train_list = []
        ds = dl_train.dataset
        for i in range(0, len(ds)):
            x_train_list.append(list(ds[i][0]))
            y_train_list.append(ds[i][1])

        n_classes = len(np.unique(np.array(y_train_list)))
        # ========================

        self.x_train = torch.Tensor(x_train_list)
        self.y_train = torch.Tensor(y_train_list)
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred_list = []
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            best_k = sorted(range(n_test), key=lambda x: dist_matrix[x, i])[:self.k]
            best_k_train = self.y_train[best_k]
            bin_count = np.bincount(best_k_train)
            my_argmax = np.argmax(bin_count)
            y_pred_list.append(my_argmax)
        y_pred = torch.as_tensor(y_pred_list, dtype=torch.int64)
        # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensi0ons are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.

    # ====== YOUR CODE: ======
    u1 = torch.sum(torch.mul(x1, x1), 1)
    u2 = torch.sum(torch.mul(x2, x2), 1)
    u1 = torch.reshape(u1, (u1.size()[0], 1))
    u2 = torch.reshape(u2, (1, u2.size()[0]))
    m = torch.mm(x1, x2.t())
    return torch.sqrt(u1 - 2 * m + u2)

    # ======== ================


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    # ====== YOUR CODE: ======
    a = y - y_pred
    b = torch.Tensor([int(i) for i in (a == 0)])
    return torch.sum(b).item() / a.size()[0]
    # ========================


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        model = KNNClassifier(k)
        local_acc = []
        for j in range(num_folds):
            train, valid = dataloaders.create_train_validation_loaders(ds_train, (1.0 / num_folds))
            model.train(train)
            acc, size = 0, 0
            for idx, (x, y) in enumerate(valid):
                acc += accuracy(y, model.predict(x))*x.size(0)
                size += x.size(0)
            local_acc.append(acc / size)
        accuracies.append(local_acc)
        # ========================
    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]
    print('best_k', best_k)
    return best_k, accuracies
