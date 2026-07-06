import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATA_YAML = Path(__file__).resolve().parents[1] / "Helmet_Detection_DataSet" / "data.yaml"


def train_yolo11s(data_yaml=DEFAULT_DATA_YAML, epochs=50, imgsz=640, batch=8):
    model = YOLO("yolo11s.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/train",
        name="helmet_yolo11s",
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11s for helmet violation detection")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    args = parser.parse_args()

    train_yolo11s(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )
