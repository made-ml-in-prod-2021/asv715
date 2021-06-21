import os
import pandas as pd
import numpy as np
import click


@click.command("preprocess")
@click.argument("input_dir")
@click.argument("output_dir")
def preprocess(input_dir: str, output_dir):
    data = pd.read_csv(
        os.path.join(input_dir, "data.csv"),
        header=0,
        index_col=0,
        dtype=np.float32,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    target = pd.read_csv(
        os.path.join(input_dir, "target.csv"),
        header=0,
        index_col=0,
        names=["target"]
    )
    data['is_sepal_length_big'] = data.apply(lambda row: 1 if row["sepal_length"] > 5 else 0, axis=1)
    data['is_sepal_width_big'] = data.apply(lambda row: 1 if row["sepal_width"] > 3 else 0, axis=1)

    data['is_petal_length_big'] = data.apply(lambda row: 1 if row["petal_length"] > 5 else 0, axis=1)
    data['is_petal_width_big'] = data.apply(lambda row: 1 if row["petal_width"] > 2 else 0, axis=1)

    data['sepal_square'] = data.apply(lambda row: row["sepal_length"] * row["sepal_width"], axis=1)
    data['petal_square'] = data.apply(lambda row: row["petal_length"] * row["petal_width"], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "train_target.csv"), index=False)


if __name__ == "__main__":
    preprocess()
