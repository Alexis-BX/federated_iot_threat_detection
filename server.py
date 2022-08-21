import flwr as fl
import joblib
import threading
from time import sleep
from sklearn.metrics import log_loss

from models import TreeClassifier, ForestClassifier, PORT, MIN_CLIENTS, ROUNDS, FOREST, SERVER_IP
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

        self.port = port
        self.rounds = rounds
        self.strategy = fl.server.strategy.FedAvg(
            min_available_clients=(MIN_CLIENTS if not FOREST else 1),
            eval_fn = self.evaluate,
            on_fit_config_fn=self.fit_round,
        )
        
        self.metrics = [None] * (MIN_CLIENTS if FOREST else 1)
        self.servers = [fl.server.Server(client_manager=fl.server.client_manager.SimpleClientManager(), strategy=self.strategy) for _ in range((MIN_CLIENTS if FOREST else 1))]

        if FOREST:
            self.strategy.min_available_clients = 1
            for i in range(MIN_CLIENTS):
                threading.Thread(target=self.launch_server, args=(i,)).start()

            # Loop to ensure every server has gotten its client
            ran = [False] * (MIN_CLIENTS if FOREST else 1)
            while sum(ran) < MIN_CLIENTS: 
                for i, server in enumerate(self.servers):
                    if not ran[i]:
                        ran[i] = len(server._client_manager.all())>0
                sleep(1) 
            
            # Stop when no more servers have clients
            while sum([len(s._client_manager.all()) for s in self.servers]):
                sleep(1)

        else:
            # Way to easy to have a single server for a simple decision tree!
            self.launch_server(0)


    def launch_server(self, i):
        self.metrics[i] = fl.server.start_server(f"{SERVER_IP}:{self.port+i}", config={"num_rounds": self.rounds}, server=self.servers[i], strategy=self.strategy)


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
    print()
    print(server.get_metrics())
    