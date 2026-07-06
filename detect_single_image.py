import argparse
from pathlib import Path

import cv2

from helmet_violation_system import (
    HelmetViolationSystem,
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAFFIC_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
)


def detect_image(
    image_path,
    confidence=0.5,
    model_path=DEFAULT_MODEL_PATH,
    traffic_model_path=DEFAULT_TRAFFIC_MODEL_PATH,
):
    system = HelmetViolationSystem(
        model_path=model_path,
        traffic_model_path=traffic_model_path,
        confidence=confidence,
    )
    annotated, detections, stats = system.process_image(image_path, confidence=confidence)

    output_dir = DEFAULT_OUTPUT_DIR / "image_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"detected_{Path(image_path).name}"
    cv2.imwrite(str(output_path), annotated)

    print(f"Processed image: {Path(image_path).resolve()}")
    print(f"Output image: {output_path}")
    print("Detections:")
    for detection in detections:
        if detection.is_traffic_object:
            status = "traffic/object"
        else:
            status = "violation" if detection.is_violation else "safe"
        print(f"  {detection.class_name}: {detection.confidence:.2f} ({status})")
    print(f"Detectable category count: {stats['supported_category_count']}")
    print(f"Traffic/object counts: {stats['traffic_counts']}")
    print(f"CSV report: {stats['report_path']}")
    print(f"Violation screenshots: {stats['violations_dir']}")
    return str(output_path), detections, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image helmet detection with evidence logging")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "pic_test.jpg"),
        help="Path to the input image",
    )
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to YOLO model weights")
    parser.add_argument(
        "--traffic-model",
        default=str(DEFAULT_TRAFFIC_MODEL_PATH),
        help="Path to COCO traffic/object YOLO model. Use yolo11s.pt for the final YOLOv11s demo.",
    )
    args = parser.parse_args()

    detect_image(
        image_path=args.image_path,
        confidence=args.confidence,
        model_path=args.model,
        traffic_model_path=args.traffic_model,
    )
