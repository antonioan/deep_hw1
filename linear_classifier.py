import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple
from .losses import ClassifierLoss

class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # DONE:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.normal(0, weight_std, (n_features, n_classes))
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # DONE:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        #biased = BiasTrick()
        #x = biased(x)
        class_scores = x.mm(self.weights)
        _, y_pred = torch.max(class_scores, 1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # DONE:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        assert y.size(0) > 0
        a = y - y_pred
        b = a.numel() - a.nonzero().size(0)
        acc = float(b) / a.size(0)
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            train_res.accuracy.append(0.)
            train_res.loss.append(0.)
            for batch in dl_train:  # batch is a tuple of inputs and true outputs
                y_pred, class_scores = self.predict(batch[0])
                train_res.accuracy[epoch_idx] += self.evaluate_accuracy(batch[1], y_pred)
                train_res.loss[epoch_idx] += loss_fn(batch[0], batch[1], class_scores, y_pred)
                self.weights = self.weights * (1 - weight_decay) - learn_rate * loss_fn.grad()
            train_res.loss[epoch_idx] /= len(dl_train)

            valid_res.accuracy.append(0.)
            valid_res.loss.append(0.)
            for batch in dl_valid:
                y_pred, class_scores = self.predict(batch[0])
                valid_res.accuracy[epoch_idx] += self.evaluate_accuracy(batch[1], y_pred)
                valid_res.loss[epoch_idx] += loss_fn(batch[0], batch[1], class_scores, y_pred)
            valid_res.loss[epoch_idx] /= len(dl_valid)
            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        w_images = self.weights[has_bias:, :].reshape((self.n_classes, *img_shape))
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0., learn_rate=0., weight_decay=0.)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['weight_std'] = 0.1
    hp['learn_rate'] = 0.05
    hp['weight_decay'] = 0.001
    # ========================

    return hp
