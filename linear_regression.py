import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.max(self.weights_ @ X, axis=1)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution
        #  Use only numpy functions. Don't forget regularization.

        w_opt = None
        # ====== YOUR CODE: ======
        y.reshape((len(y), 1))
        print(y.shape)
        print((X.T).shape)
        a = y @ X.T
        m = a @ np.linalg.inv((np.eye(X.shape)) + X @ X.T)
        w_opt = np.dot(m, len(y)/2)
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # DONE:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        xb = np.hstack((np.ones((X.shape[0], 1)), X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    target_mean = np.mean(df[target_feature])
    target_var = np.sqrt(np.sum((df[target_feature] - target_mean * np.ones_like(df[target_feature])) ** 2))
    # target_vec = (df[target_feature] - target_mean * np.ones_like(target_mean)).T
    target_vec = (df[target_feature] - target_mean * np.ones_like(target_mean))
    pearson = []

    for feature_name, feature in df.iteritems():
        if feature_name == target_feature:
            continue
        feature_mean = np.mean(feature)
        feature_vec = feature - feature_mean * np.ones_like(feature)
        feature_var = np.sqrt(np.sum((feature - feature_mean * np.ones_like(feature)) ** 2))
        covar = np.matmul(feature_vec.T, target_vec)
        pearson.append([np.abs(covar / (target_var * feature_var)), feature_name, covar / (target_var * feature_var)])

    pearson.sort(key=lambda x: float(x[0]), reverse=True)
    pearson = pearson[:n]
    top_n_features = [item[1] for item in pearson]
    top_n_corr = [item[2] for item in pearson]
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    sum_squares = np.sum(np.square(np.subtract(y, y_pred)))
    mse = (sum_squares / len(y))  # / 2
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    e = np.subtract(y, y_pred)
    y_bar = np.sum(y) / len(y)
    y_bar_vec = np.full((len(y), ), y_bar)
    numerator = np.sum(np.square(e))
    denominator = np.sum(np.square(np.subtract(y, y_bar_vec)))
    r2 = 1 - numerator / denominator
    # ========================
    return r2


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    return best_params
