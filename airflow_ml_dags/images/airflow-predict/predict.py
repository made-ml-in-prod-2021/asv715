import os
import pandas as pd
import click
import pickle


@click.command("predict")
@click.argument("input_dir")
@click.argument("model_dir")
@click.argument("output_dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))

    with open(model_dir + "/model.pkl", "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir + "/predictions.csv", "w") as file:
        for item in predictions:
            file.write(str(item) + "\n")


if __name__ == "__main__":
    predict()
