import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from data_handle import request_data_server, get_1_all


def fit_round(rnd):
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load test data
    _, X_test, _, y_test = request_data_server()

    # evaluate function called after every round
    def evaluate(parameters):
        global model
        # Update model with the latest parameters
        model = utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = utils.generate_model()
    
    X_init, y_init = get_1_all()
    model.fit(X_init, y_init)

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn = get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    loss = fl.server.start_server("0.0.0.0:5000", strategy=strategy, config={"num_rounds": 5})
    
    print("EEEEEEEEEEEEEEND", loss)