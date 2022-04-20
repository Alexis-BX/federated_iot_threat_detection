import warnings
import flwr as fl
import sys
from sklearn.metrics import log_loss

from models import FOREST, TreeClassifier, PORT
from data_handle import request_data_client

class MnistClient(fl.client.NumPyClient):
    def __init__(self, number):
        self.model = TreeClassifier(number)
        
        (self.X_train, self.y_train), (self.X_test, self.y_test), (X_init, y_init) = request_data_client(number)

        # give the model the correct shape
        self.model.fit(X_init, y_init)
        
    def get_parameters(self): 
        return self.model.get_params()

    def fit(self, parameters, config):
        self.model.set_model_params(parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        
        print(f"Training finished for round {config['rnd']}")
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_model_params(parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        # print(accuracy)
        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    # know which client it is to grab the correct data partition
    number = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    
    # Start Flower client
    PORT += number if FOREST else 0

    fl.client.start_numpy_client(f"0.0.0.0:{PORT}", client=MnistClient(number))
    