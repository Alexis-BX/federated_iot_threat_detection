import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# from sklearn.linear_model import LogisticRegression

PORT = 5000
MIN_CLIENTS = 2
ROUNDS = 1

def generate_model():
    """ Used to repidely change the model to use """
    #return TreeClassifier()
    return ForestClassifier(MIN_CLIENTS)

class TreeClassifier(DecisionTreeClassifier):
    def __init__(self):
        super().__init__(
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
        self.first = True
        self.name = 'decisionTree'
        self.id = id

    def set_model_params(self, params):
        if self.first:
            self.first = False
            self.params_init(params)
        else:
            self.params_update(params)

    def params_init(self, params):
        for idx, (i, j, k) in enumerate(zip(params[0],params[1],params[2])):
            self.tree_.feature[idx] = i
            self.tree_.threshold[idx] = j
            self.tree_.value[idx] = k

    def params_update(self, params):
        for idx, (i, j, k) in enumerate(zip(params[0],params[1],params[2])):
            self.tree_.feature[idx] = (self.tree_.feature[idx] + i) / 2
            self.tree_.threshold[idx] = (self.tree_.threshold[idx] + j) / 2
            self.tree_.value[idx] = (self.tree_.value[idx] + k) / 2

    def get_params(self):
        """Returns the paramters of a sklearn LogisticRegression model."""
        return [self.tree_.feature, self.tree_.threshold, self.tree_.value]

class TreeForestClassifier(DecisionTreeClassifier):
    def __init__(self, id):
        super().__init__(
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
        self.first = True
        self.name = 'decisionTreeForest'
        self.id = id

    def set_model_params(self, params):
        return

    def get_params(self):
        """Returns the paramters of a sklearn LogisticRegression model."""
        print(self.id)
        return [np.array([self.id]), self.tree_.feature, self.tree_.threshold, self.tree_.value]

class ForestClassifier:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = [TreeClassifier() for _ in range(n_trees)]
        self.first = True
        self.name = 'randomForest'

    def fit(self, X, y):
        for i in range(self.n_trees):
            self.trees[i].fit(X, y)

    def most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self.most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

    def set_model_params(self, params):
        if self.first:
            self.first = False
            self.params_init(params)
        else:
            self.params_update(params)

    def params_init(self, params):
        t, params = params[0][0], params[1:]
        
        for idx, (i, j, k) in enumerate(zip(params[0], params[1], params[2])):
            self.trees[t].tree_.feature[idx] = i
            self.trees[t].tree_.threshold[idx] = j
            self.trees[t].tree_.value[idx] = k

    def params_update(self, params):
        t, params = params[0][0], params[1:]
        print(t)
        for idx, (i, j, k) in enumerate(zip(params[0], params[1], params[2])):
            self.trees[t].tree_.feature[idx] = (self.trees[t].tree_.feature[idx] + i) / 2
            self.trees[t].tree_.threshold[idx] = (self.trees[t].tree_.threshold[idx] + j) / 2
            self.trees[t].tree_.value[idx] = (self.trees[t].tree_.value[idx] + k) / 2

    def get_params(self):
        """Returns the paramters of a sklearn LogisticRegression model."""
        return []

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# class ForestClassifier(RandomForestClassifier):
#     def __init__(self):
#         super().__init__(
#             n_estimators=5, 
#             criterion='gini', 
#             max_depth=3,
#             warm_start=True
#         )
#         self.first = True
#         self.name = 'randomForest'

#     def set_model_params(self, params):
#         if self.first:
#             self.first = False
#             self.params_init(params)
#         else:
#             self.params_update(params)

#     def params_init(self, params):
#         for t in range(self.n_estimators):
#             print(self.estimators_[t].get_depth())
#             for idx, (i, j, k) in enumerate(zip(params[3*t+0],params[3*t+1],params[3*t+2])):
#                 self.estimators_[t].tree_.feature[idx] = i
#                 self.estimators_[t].tree_.threshold[idx] = j
#                 self.estimators_[t].tree_.value[idx] = k

#     def params_update(self, params):
#         for t in range(self.n_estimators):
#             print(self.estimators_[t].get_depth())
#             for idx, (i, j, k) in enumerate(zip(params[3*t+0],params[3*t+1],params[3*t+2])):
#                 self.estimators_[t].tree_.feature[idx] = (self.estimators_[t].tree_.feature[idx] + i) / 2
#                 self.estimators_[t].tree_.threshold[idx] = (self.estimators_[t].tree_.threshold[idx] + j) / 2
#                 self.estimators_[t].tree_.value[idx] = (self.estimators_[t].tree_.value[idx] + k) / 2

#     def get_params(self):
#         """Returns the paramters of a sklearn LogisticRegression model."""
#         return sum([self.get_params_tree(self.estimators_[i]) for i in range(self.n_estimators)], [])

#     def get_params_tree(self, tree):
#         """Returns the paramters of a sklearn LogisticRegression model."""
#         return [tree.tree_.feature, tree.tree_.threshold, tree.tree_.value]
