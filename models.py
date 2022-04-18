import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# from sklearn.linear_model import LogisticRegression

PORT = 5000
MIN_CLIENTS = 2
ROUNDS = 5

def generate_model():
    """ Used to repidely change the model to use """
    return TreeClassifier()

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

################
# class ForestClassifier(RandomForestClassifier):
#     def __init__(self):
#         super().__init__(
#             n_estimators=5, 
#             criterion='gini', 
#             max_depth=3,
#         )
#         self.first = True

#     def set_model_params(self, params):
#         if self.first:
#             self.first = False
#             self.params_init(params)
#         else:
#             self.params_update(params)

#     def params_init(self, params):
#         for idx, (i, j, k) in enumerate(zip(params[0],params[1],params[2])):
#             self.tree_.feature[idx] = i
#             self.tree_.threshold[idx] = j
#             self.tree_.value[idx] = k

#     def params_update(self, params):
#         for idx, (i, j, k) in enumerate(zip(params[0],params[1],params[2])):
#             self.tree_.feature[idx] = (self.tree_.feature[idx] + i) / 2
#             self.tree_.threshold[idx] = (self.tree_.threshold[idx] + j) / 2
#             self.tree_.value[idx] = (self.tree_.value[idx] + k) / 2

#     def get_params(self):
#         """Returns the paramters of a sklearn LogisticRegression model."""
#         return [self.tree_.feature, self.tree_.threshold, self.tree_.value]