"""
DNA/microbiome sequence classifier (DNN) with a command-line interface.

Usage examples
--------------
    python dna_dnn_cli.py train --input data.csv --epochs 5 --lr 1e-3
    python dna_dnn_cli.py train --input data.csv --seq-col sequences --label-col label \
        --unknown-col unknown --output-preds preds.csv --plot-out roc.png --model-out model.pt
    python dna_dnn_cli.py --help
"""

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import BCEWithLogitsLossFlat
from fastai.metrics import RocAucBinary, accuracy
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")  # safe for headless CLI runs
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

BASE_MAPPING = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}


def one_hot_encode(sequence: str) -> np.ndarray:
    return np.array([BASE_MAPPING[base] for base in sequence], dtype=np.float32)


def encode_sequences(sequences) -> np.ndarray:
    return np.array([one_hot_encode(seq).flatten() for seq in sequences], dtype=np.float32)


class DNADNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


def run_pipeline(
    input_csv: str,
    seq_col: str,
    label_col: str,
    unknown_col: str,
    epochs: int,
    lr: float,
    batch_size: int,
    test_size: float,
    seed: int,
    preds_out: str,
    plot_out: str,
    model_out: str,
    show_plot: bool,
):
    click.echo(f"Loading data from {input_csv} ...")
    datalabel = pd.read_csv(input_csv)

    sequences = datalabel[seq_col].to_list()
    labels = datalabel[label_col].to_list()

    has_unknown = unknown_col in datalabel.columns
    X_new = datalabel[unknown_col].to_list() if has_unknown else []

    click.echo("One-hot encoding sequences ...")
    X = encode_sequences(sequences)
    y = np.array(labels, dtype=np.float32)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    X_train_t = torch.tensor(X_train)
    X_valid_t = torch.tensor(X_valid)
    y_train_t = torch.tensor(y_train)
    y_valid_t = torch.tensor(y_valid)

    train_ds = TensorDataset(X_train_t, y_train_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

    dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=batch_size)
    xb, yb = dls.one_batch()
    click.echo(f"Batch shapes -> X: {tuple(xb.shape)}, y: {tuple(yb.shape)}")

    input_size = X_train_t.shape[1]
    model = DNADNN(input_size)

    learner = Learner(
        dls,
        model,
        loss_func=BCEWithLogitsLossFlat(),
        metrics=[accuracy, RocAucBinary()],
    )

    click.echo(f"Training for {epochs} epochs at lr={lr} ...")
    learner.fit(epochs, lr=lr)
    learner.validate()

    fig = learner.recorder.plot_loss()
    loss_plot_path = "loss_curve.png"
    plt.savefig(loss_plot_path)
    click.echo(f"Saved loss curve to {loss_plot_path}")
    plt.close("all")

    preds, targets = learner.get_preds(dl=learner.dls.valid)
    probabilities = torch.sigmoid(preds.squeeze())
    predicted = (probabilities >= 0.5).int()
    actual = targets.int()

    cm = confusion_matrix(actual, predicted)
    click.echo(f"Confusion matrix:\n{cm}")

    fpr, tpr, _ = roc_curve(actual.numpy(), probabilities.numpy())
    auc = roc_auc_score(actual.numpy(), probabilities.numpy())
    click.echo(f"ROC-AUC: {auc:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"DNA DNN ROC Curve (AUC = {auc:.3f})")
    plt.savefig(plot_out)
    click.echo(f"Saved ROC curve to {plot_out}")
    if show_plot:
        plt.show()
    plt.close("all")

    if model_out:
        torch.save(model.state_dict(), model_out)
        click.echo(f"Saved model weights to {model_out}")

    if has_unknown and len(X_new) > 0:
        click.echo(f"Scoring {len(X_new)} unknown sequences from column '{unknown_col}' ...")
        X_tensor = torch.tensor(encode_sequences(X_new), dtype=torch.float32)
        new_ds = TensorDataset(X_tensor)
        new_dl = DataLoader(new_ds, batch_size=batch_size)

        learner.model.eval()
        preds_list = []
        with torch.no_grad():
            for (xb,) in new_dl:
                logits = learner.model(xb)
                probs = torch.sigmoid(logits.squeeze(1))
                preds_list.append(probs)

        probabilities_new = torch.cat(preds_list)
        predicted_labels = (probabilities_new >= 0.5).long()

        results = pd.DataFrame(
            {
                "sequence": X_new,
                "probability": probabilities_new.numpy(),
                "prediction": predicted_labels.numpy(),
            }
        )
        results.to_csv(preds_out, index=False)
        click.echo(f"Saved predictions to {preds_out}")
        click.echo(results.head())
    else:
        click.echo(f"No '{unknown_col}' column found (or empty) — skipping prediction on new sequences.")

    return {"auc": auc, "confusion_matrix": cm}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@click.group()
def cli():
    """DNA / microbiome sequence classifier CLI."""
    pass


@cli.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True),
              help="Path to input CSV file.")
@click.option("--seq-col", default="sequences", show_default=True,
              help="Column name containing DNA sequences.")
@click.option("--label-col", default="label", show_default=True,
              help="Column name containing binary labels.")
@click.option("--unknown-col", default="unknown", show_default=True,
              help="Column name containing sequences to predict on (optional).")
@click.option("--epochs", default=5, show_default=True, type=int,
              help="Number of training epochs.")
@click.option("--lr", default=1e-3, show_default=True, type=float,
              help="Learning rate.")
@click.option("--batch-size", default=64, show_default=True, type=int,
              help="Batch size for train/valid dataloaders.")
@click.option("--test-size", default=0.2, show_default=True, type=float,
              help="Fraction of data used for validation split.")
@click.option("--seed", default=42, show_default=True, type=int,
              help="Random seed for the train/valid split.")
@click.option("--preds-out", default="predictions.csv", show_default=True,
              help="Where to save predictions on unknown sequences.")
@click.option("--plot-out", default="roc_curve.png", show_default=True,
              help="Where to save the ROC curve plot.")
@click.option("--model-out", default="dna_dnn_model.pt", show_default=True,
              help="Where to save trained model weights (set to '' to skip saving).")
@click.option("--show-plot/--no-show-plot", default=False,
              help="Whether to display plots interactively (requires a GUI backend).")
def train(input_csv, seq_col, label_col, unknown_col, epochs, lr, batch_size,
          test_size, seed, preds_out, plot_out, model_out, show_plot):
    """Train the DNN on labeled sequences and (optionally) predict unknowns."""
    run_pipeline(
        input_csv=input_csv,
        seq_col=seq_col,
        label_col=label_col,
        unknown_col=unknown_col,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        preds_out=preds_out,
        plot_out=plot_out,
        model_out=model_out,
        show_plot=show_plot,
    )


@cli.command()
@click.option("--input", "input_csv", required=True, type=click.Path(exists=True),
              help="Path to CSV containing sequences to classify.")
@click.option("--seq-col", default="sequences", show_default=True,
              help="Column name containing DNA sequences.")
@click.option("--model-in", required=True, type=click.Path(exists=True),
              help="Path to saved model weights (.pt) to load.")
@click.option("--preds-out", default="predictions.csv", show_default=True,
              help="Where to save predictions.")
@click.option("--batch-size", default=64, show_default=True, type=int,
              help="Batch size for inference.")
def predict(input_csv, seq_col, model_in, preds_out, batch_size):
    """Load a trained model and run predictions on a CSV of sequences (no retraining)."""
    df = pd.read_csv(input_csv)
    sequences = df[seq_col].to_list()

    X_tensor = torch.tensor(encode_sequences(sequences), dtype=torch.float32)
    input_size = X_tensor.shape[1]

    model = DNADNN(input_size)
    model.load_state_dict(torch.load(model_in, map_location="cpu"))
    model.eval()

    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=batch_size)

    preds_list = []
    with torch.no_grad():
        for (xb,) in dl:
            logits = model(xb)
            probs = torch.sigmoid(logits.squeeze(1))
            preds_list.append(probs)

    probabilities = torch.cat(preds_list)
    predicted_labels = (probabilities >= 0.5).long()

    results = pd.DataFrame(
        {
            "sequence": sequences,
            "probability": probabilities.numpy(),
            "prediction": predicted_labels.numpy(),
        }
    )
    results.to_csv(preds_out, index=False)
    click.echo(f"Saved predictions to {preds_out}")
    click.echo(results.head())


if __name__ == "__main__":
    cli()