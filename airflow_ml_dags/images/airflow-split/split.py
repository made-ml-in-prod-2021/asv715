import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split")
@click.argument("input_dir")
@click.argument("output_dir")
def split(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))

    train_data, val_data, train_target, val_target = train_test_split(
        data, target, test_size=0.3, random_state=42
    )

    os.makedirs(output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)

    train_target.to_csv(os.path.join(output_dir, "train_target.csv"), index=False)
    val_target.to_csv(os.path.join(output_dir, "val_target.csv"), index=False)


if __name__ == "__main__":
    split()
