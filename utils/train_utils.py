from utils.callbacks import CallbackManager

from tqdm import tqdm

import torch


def train_one_epoch(

    model,

    dataloader,

    optimizer,

    criterion,

    metrics,

    device,

):

    model.train()

    running_loss = 0.0

    running_metrics = {

        name: 0.0

        for name in metrics.keys()

    }

    progress = tqdm(

        dataloader,

        leave=False,

    )

    for images, masks in progress:

        images = images.to(device)

        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(

            outputs,

            masks,

        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():

            for name, metric in metrics.items():

                value = metric(

                    outputs,

                    masks,

                )

                if isinstance(value, torch.Tensor):
                    value = value.item()

                running_metrics[name] += value

    epoch_loss = running_loss / len(dataloader)

    epoch_metrics = {

        name: value / len(dataloader)

        for name, value in running_metrics.items()

    }

    return epoch_loss, epoch_metrics


def validate(

    model,

    dataloader,

    criterion,

    metrics,

    device,

):

    model.eval()

    running_loss = 0.0

    running_metrics = {

        name: 0.0

        for name in metrics.keys()

    }

    with torch.no_grad():

        for images, masks in dataloader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(

                outputs,

                masks,

            )

            running_loss += loss.item()

            for name, metric in metrics.items():

                value = metric(

                    outputs,

                    masks,

                )

                if isinstance(value, torch.Tensor):
                    value = value.item()

                running_metrics[name] += value

    epoch_loss = running_loss / len(dataloader)

    epoch_metrics = {

        name: value / len(dataloader)

        for name, value in running_metrics.items()

    }

    return epoch_loss, epoch_metrics


def train_model(

    model,

    train_loader,

    val_loader,

    optimizer,

    criterion,

    metrics,

    device,

    epochs,

    model_name,

):

    callback = CallbackManager(

        model_name=model_name,

        optimizer=optimizer,

    )

    model.to(device)

    for epoch in range(epochs):

        print(

            f"\nEpoch "

            f"{epoch + 1}/{epochs}"

        )

        train_loss, train_metrics = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            metrics,

            device,

        )

        val_loss, val_metrics = validate(

            model,

            val_loader,

            criterion,

            metrics,

            device,

        )

        print(

            f"Train Loss: {train_loss:.4f} | "

            f"Val Loss: {val_loss:.4f}"

        )

        print()

        for name in metrics.keys():

            print(

                f"{name:10}"

                f" Train: {train_metrics[name]:.4f}"

                f" | "

                f"Val: {val_metrics[name]:.4f}"

            )

        callback.step(

            model=model,

            epoch=epoch + 1,

            train_loss=train_loss,

            val_loss=val_loss,

            train_metrics=train_metrics,

            val_metrics=val_metrics,

        )

        if callback.stop:

            break

    print("\nTraining Finished.")