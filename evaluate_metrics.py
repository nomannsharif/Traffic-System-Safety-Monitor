import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from helmet_violation_system import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR


DEFAULT_DATA_YAML = Path(__file__).resolve().parents[1] / "Helmet_Detection_DataSet" / "data.yaml"


def evaluate_model(model_path=DEFAULT_MODEL_PATH, data_yaml=DEFAULT_DATA_YAML, split="test", imgsz=640):
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        verbose=False,
    )

    results = metrics.results_dict
    report = {
        "split": split,
        "precision": results.get("metrics/precision(B)", 0.0),
        "recall": results.get("metrics/recall(B)", 0.0),
        "mAP50": results.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": results.get("metrics/mAP50-95(B)", 0.0),
        "fitness": results.get("fitness", 0.0),
    }

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DEFAULT_OUTPUT_DIR / f"evaluation_metrics_{split}.csv"
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=report.keys())
        writer.writeheader()
        writer.writerow(report)

    return report_path, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate helmet detection model metrics")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to YOLO model weights")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    args = parser.parse_args()

    output_path, report = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
    )

    print(f"Metrics saved to: {output_path}")
    for key, value in report.items():
        print(f"{key}: {value}")
