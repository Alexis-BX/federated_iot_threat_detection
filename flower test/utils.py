import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

FIRST = True

def generate_model():
    # used to repidely change the model to use
    return generate_tree()

def generate_tree():
    return DecisionTreeClassifier(
        ccp_alpha=0.0, 
        class_weight={i: 1 for i in range(2)}, 
        criterion='gini', 
        max_depth=3, 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        min_samples_leaf=1,
        min_samples_split=2, 
        min_weight_fraction_leaf=0.0, 
        random_state=None,
        splitter='best'
    )

def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model."""
    t = model.tree_
    return [t.feature, t.threshold, t.value]


def set_model_params(model, params):
    """Sets the parameters of a sklean LogisticRegression model."""
    global FIRST
    if FIRST:
        for idx, (i, j, k) in enumerate(zip(params[0],params[1],params[2])):
            model.tree_.feature[idx] = i
            model.tree_.threshold[idx] = j
            model.tree_.value[idx] = k
        FIRST = False
    else:
        for idx, (i, j, k) in enumerate(zip(params[0],params[1],params[2])):
            model.tree_.feature[idx] = (model.tree_.feature[idx] + i) / 2
            model.tree_.threshold[idx] = (model.tree_.threshold[idx] + j) / 2
            model.tree_.value[idx] = (model.tree_.value[idx] + k) / 2

    #if model.fit_intercept:
    #    model.intercept_ = params[1]
    return model


def set_initial_params(model):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  # MNIST has 10 classes
    n_features = 6  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    #if model.fit_intercept:
    #    model.intercept_ = np.zeros((n_classes,))
    return model
