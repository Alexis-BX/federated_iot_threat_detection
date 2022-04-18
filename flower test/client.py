import warnings
import flwr as fl
import numpy as np
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils
from data_handle import request_data_client, get_1_all

if __name__ == "__main__":
    model = utils.generate_model()
    X_init, y_init = get_1_all()
    model.fit(X_init, y_init)

    # know which client it is to grab the correct data partition
    number = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    X_train, X_test, y_train, y_test = request_data_client(number)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self): 
            return utils.get_model_parameters(model)

        def fit(self, parameters, config): 
            global model
            model = utils.set_model_params(model, parameters)

            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            global model
            model = utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            # print(accuracy)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:5000", client=MnistClient())
