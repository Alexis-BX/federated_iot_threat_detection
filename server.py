import flwr as fl
import joblib
from sklearn.metrics import log_loss

from models import generate_model, PORT
from data_handle import request_data_server

class Server:
    def __init__(self, rounds=5, port=5000):
        self.model = generate_model()
    
        _, (self.X_test, self.y_test), (X_init, y_init) = request_data_server()
        
        # give the model the correct shape
        self.model.fit(X_init, y_init)

        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            eval_fn = self.evaluate,
            on_fit_config_fn=self.fit_round,
        )
        
        self.metrics = fl.server.start_server(f"0.0.0.0:{port}", strategy=strategy, config={"num_rounds": rounds})

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
    server = Server(rounds=5, port=PORT)
    joblib.dump(server.model, 'tree.joblib')
    # model = load('tree.joblib')

    print()
    print(server.metrics)