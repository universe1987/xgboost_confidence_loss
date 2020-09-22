import os
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_color_codes()


class XGBConfidenceClassifier(XGBRegressor):
    """
    Binary classifier with confidence loss.
    This only works for binary classification, I customized the loss function to be equivalent to the confidence loss.
    """

    def __init__(self, **kwargs):
        super().__init__(objective=XGBConfidenceClassifier.confidence_loss,
                         max_depth=5, learning_rate=0.1, n_estimators=100,
                         verbosity=None, booster='gbtree',
                         tree_method=None, n_jobs=1, gamma=0,
                         min_child_weight=1, max_delta_step=None, subsample=1,
                         colsample_bytree=1, colsample_bylevel=1,
                         colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                         scale_pos_weight=1, base_score=0.5, random_state=None,
                         missing=np.nan, num_parallel_tree=None,
                         monotone_constraints=None, interaction_constraints=None,
                         importance_type="gain", gpu_id=None,
                         validate_parameters=None, **kwargs)

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        return self

    def predict(self, X, output_margin=False, ntree_limit=None, validate_features=True, base_margin=None):
        y = super().predict(X)
        return 1 - 1 / (1 + np.exp(-y))

    def predict_proba(self, X, ntree_limit=None, validate_features=True, base_margin=None):
        y = self.predict(X)
        return np.array([y, 1 - y]).T

    @staticmethod
    def confidence_loss(y_true, y_pred):
        prob = 1.0 / (1.0 + np.exp(-y_pred))
        grad = prob - y_true
        hess = prob * (1 - prob)
        # Hessian of the KL term is the same as Hessian of binary cross entropy
        grad_ood = prob - 0.5
        grad[y_true == -1] = grad_ood[y_true == -1]
        return grad, hess


def generate_training_data(n_negative=1000, n_positive=1000, n_ood=1000):
    """
    Generate training data with 3 classes {0, 1, -1,} OOD data are labeled -1.
    """
    n = n_negative + n_positive + n_ood
    x = np.zeros([n, 2])
    y = np.zeros(n)
    x[:n_negative, 0] = np.random.normal(25, 2, n_negative)
    x[:n_negative, 1] = np.random.normal(25, 2, n_negative)
    x[n_negative:n_negative + n_positive, 0] = np.random.normal(30, 2, n_positive)
    x[n_negative:n_negative + n_positive, 1] = np.random.normal(30, 2, n_positive)
    y[n_negative:n_negative + n_positive] = 1
    y[n_negative + n_positive:] = -1
    x[n_negative + n_positive:, 0] = np.random.uniform(0, 50, n_ood)
    x[n_negative + n_positive:, 1] = np.random.uniform(0, 50, n_ood)
    return x, y


def train_confidence_classifier(x, y):
    clf = XGBConfidenceClassifier()
    clf.fit(x, y)
    return clf


def train_binary_classifier(x, y):
    clf = XGBClassifier()
    clf.fit(x[y >= 0], y[y >= 0])
    return clf


def train_multi_class_classifier(x, y):
    clf = XGBClassifier()
    # change OOD label to 2 in order to fit multi class classifier
    y_copy = y.copy()
    y_copy[y_copy < 0] = 2
    clf.fit(x, y_copy)
    return clf


def plot_heat_map(prediction, filename):
    sns.heatmap(prediction)
    fig = plt.gcf()
    fig.set_size_inches(14, 12)
    plt.savefig(filename)
    plt.close()


def plot_decision_boundaries(x, prediction, threshold0, threshold1, filename):
    group0 = prediction <= threshold0
    plt.scatter(x[group0, 0], x[group0, 1], s=5, color='r')
    group1 = prediction >= threshold1
    plt.scatter(x[group1, 0], x[group1, 1], s=5, color='g')
    group_ood = (threshold0 < prediction) & (prediction < threshold1)
    plt.scatter(x[group_ood, 0], x[group_ood, 1], s=5, color='b')
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.savefig(filename)
    plt.close()


def plot_data(x, y):
    x0 = x[y == 0]
    plt.scatter(x0[:, 0], x0[:, 1], s=5, color='r')
    x1 = x[y == 1]
    plt.scatter(x1[:, 0], x1[:, 1], s=5, color='g')
    plt.xlim([0, 50])
    plt.ylim([0, 50])
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    plt.savefig('data/data.png')
    plt.close()


def test_classifiers():
    train_x, train_y = generate_training_data()
    plot_data(train_x, train_y)

    binary_classifier = train_binary_classifier(train_x, train_y)
    multi_class_classifier = train_multi_class_classifier(train_x, train_y)
    confidence_classifier = train_confidence_classifier(train_x, train_y)

    x = np.array(np.meshgrid(np.linspace(0, 50, 200), np.linspace(0, 50, 200))).reshape(2, -1).T
    p = binary_classifier.predict_proba(x)
    plot_decision_boundaries(x, p[:, 0], 0.2, 0.8, 'data/boundary_binary.png')
    pred = p[:, 0].reshape(200, 200)[::-1, :]
    plot_heat_map(pred, 'data/heatmap_binary.png')

    p = multi_class_classifier.predict_proba(x)
    pp = p[:, 0].copy()
    pp[p[:, 2] > 0.8] = 0.5
    pp[p[:, 1] > 0.8] = 0
    plot_decision_boundaries(x, pp, 0.2, 0.8, 'data/boundary_multi_class.png')
    p = p[:, 0] - p[:, 1]
    p = p - p.min()
    p = p / p.max()
    pred = p.reshape(200, 200)[::-1, :]
    plot_heat_map(pred, 'data/heatmap_multi_class.png')

    p = confidence_classifier.predict_proba(x)
    plot_decision_boundaries(x, p[:, 0], 0.2, 0.8, 'data/boundary_confidence_loss.png')
    pred = p[:, 0].reshape(200, 200)[::-1, :]
    plot_heat_map(pred, 'data/heatmap_confidence_loss.png')


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    test_classifiers()
