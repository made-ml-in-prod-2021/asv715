import os
import click
from sklearn.datasets import load_iris


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    data, target = load_iris(return_X_y=True, as_frame=True)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"))
    target.to_csv(os.path.join(output_dir, "target.csv"))


if __name__ == '__main__':
    download()