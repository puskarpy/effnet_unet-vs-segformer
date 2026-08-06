from pathlib import Path
import csv


def save_metrics(

    model_name,

    dice,

    iou,

    precision,

    recall,

    inference_time,

    save_path="results/metrics.csv",

):

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = save_path.exists()

    with open(
        save_path,
        "a",
        newline="",
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "Model",
                "Dice",
                "IoU",
                "Precision",
                "Recall",
                "Inference Time (ms)",
            ])

        writer.writerow([
            model_name,
            round(dice, 4),
            round(iou, 4),
            round(precision, 4),
            round(recall, 4),
            round(inference_time, 2),
        ])