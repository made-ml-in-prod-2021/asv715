import os
import pandas as pd
import click
import pickle
from sklearn.linear_model import LogisticRegression


@click.command("fit")
@click.argument("input_dir")
@click.argument("output_dir")
def fit(input_dir: str, output_dir):
    train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    train_target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))

    model = LogisticRegression()
    model.fit(train_data, train_target)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir + "/model.pkl", "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    fit()
