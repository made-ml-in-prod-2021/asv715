import os
import pandas as pd
import click
import json
import pickle
from sklearn.metrics import accuracy_score


@click.command("validate")
@click.argument("input_dir")
@click.argument("model_dir")
@click.argument("output_dir")
def validate(input_dir: str, model_dir: str, output_dir):
    val_data = pd.read_csv(os.path.join(input_dir, "val_data.csv"))
    val_target = pd.read_csv(os.path.join(input_dir, "val_target.csv"))

    with open(model_dir + "/model.pkl", "rb") as file:
        model = pickle.load(file)

    predict = model.predict(val_data)
    metrics = {
        "accuracy": accuracy_score(val_target, predict)
    }

    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir + "/metrics.json", "w") as file:
        dumped = json.dumps(metrics)
        json.dump(dumped, file)


if __name__ == "__main__":
    validate()
