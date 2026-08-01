from pathlib import Path

import pandas as pd
import torch


class CallbackManager:

    def __init__(

        self,

        model_name,

        optimizer,

        patience=10,

        factor=0.5,

        min_lr=1e-6,

        checkpoint_dir="checkpoints",

    ):

        self.optimizer = optimizer

        self.patience = patience

        self.best_loss = float("inf")

        self.counter = 0

        self.stop = False

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.best_model_path = (

            self.checkpoint_dir /

            f"best_{model_name}.pth"

        )

        self.last_model_path = (

            self.checkpoint_dir /

            f"last_{model_name}.pth"

        )

        self.history_path = (

            self.checkpoint_dir /

            f"{model_name}_history.csv"

        )

        self.history = []

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=factor,

            patience=5,

            min_lr=min_lr,

        )

    def step(

        self,

        model,

        epoch,

        train_loss,

        val_loss,

        train_metrics,

        val_metrics,

    ):

        # -----------------------------
        # Save history
        # -----------------------------

        row = {

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": val_loss,

        }

        for name, value in train_metrics.items():

            row[f"train_{name}"] = value

        for name, value in val_metrics.items():

            row[f"val_{name}"] = value

        self.history.append(row)

        pd.DataFrame(self.history).to_csv(

            self.history_path,

            index=False,

        )

        # -----------------------------
        # LR Scheduler
        # -----------------------------

        self.scheduler.step(val_loss)

        # -----------------------------
        # Save last model
        # -----------------------------

        torch.save(

            model.state_dict(),

            self.last_model_path,

        )

        # -----------------------------
        # Save best model
        # -----------------------------

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            self.counter = 0

            torch.save(

                model.state_dict(),

                self.best_model_path,

            )

            print(

                f"\n✓ Validation improved "

                f"({val_loss:.4f})"

            )

        else:

            self.counter += 1

            print(

                f"\nNo improvement "

                f"({self.counter}/{self.patience})"

            )

        # -----------------------------
        # Early stopping
        # -----------------------------

        if self.counter >= self.patience:

            self.stop = True

            print(

                "\nEarly stopping triggered."

            )