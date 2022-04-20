import flwr as fl
import joblib
from sklearn.metrics import log_loss

from models import TreeClassifier, ForestClassifier, PORT, MIN_CLIENTS, ROUNDS, FOREST
from data_handle import request_data_server

class Server:
    def __init__(self, rounds=5, port=5000):
        if FOREST:
            self.model = ForestClassifier(MIN_CLIENTS)
        else:
            self.model = TreeClassifier()
    
        (X_train, y_train), (self.X_test, self.y_test), (X_init, y_init) = request_data_server()
        
        # give the model the correct shape
        self.model.fit(X_init, y_init)

        strategy = fl.server.strategy.FedAvg(
            min_available_clients=MIN_CLIENTS,
            eval_fn = self.evaluate,
            on_fit_config_fn=self.fit_round,
        )
        
        if FOREST:
            strategy.min_available_clients = 1
            self.metrics = []
            for i in range(MIN_CLIENTS):
                self.metrics.append(fl.server.start_server(f"0.0.0.0:{port+i}", strategy=strategy, config={"num_rounds": rounds}))
        else:
            self.metrics = fl.server.start_server(f"0.0.0.0:{port}", strategy=strategy, config={"num_rounds": rounds})
        
    def get_metrics(self):
        return self.model.score(self.X_test, self.y_test)

    def fit_round(self, rnd):
        """Send round number to client."""
        return {"rnd": rnd}

    # evaluate function called after every round
    def evaluate(self, parameters):
        # Update model with the latest parameters
        self.model.set_model_params(parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, {"accuracy": accuracy}


if __name__ == "__main__":
    # Start Flower server for five rounds of federated learning
    server = Server(rounds=ROUNDS, port=PORT)
    joblib.dump(server.model, server.model.name+'.joblib')
    # model = load('tree.joblib')

    print()
    print(server.metrics)
    