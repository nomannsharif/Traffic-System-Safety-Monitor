import argparse

from helmet_violation_system import HelmetViolationSystem, DEFAULT_MODEL_PATH, DEFAULT_TRAFFIC_MODEL_PATH


def detect_webcam(
    confidence=0.5,
    camera_index=0,
    model_path=DEFAULT_MODEL_PATH,
    traffic_model_path=DEFAULT_TRAFFIC_MODEL_PATH,
):
    system = HelmetViolationSystem(
        model_path=model_path,
        traffic_model_path=traffic_model_path,
        confidence=confidence,
    )

    print("Starting webcam violation tracking")
    print("Press q in the OpenCV window to stop")
    stats = system.process_webcam(
        camera_index=camera_index,
        confidence=confidence,
    )

    print("Webcam detection stopped")
    print(f"Frames: {stats['frames']}")
    print(f"Unique violations: {stats['unique_violations']}")
    print(f"Raw violation detections: {stats['raw_violation_detections']}")
    print(f"Duplicate count reduction: {stats['duplicate_count_reduction']}")
    print(f"Safe detections: {stats['safe_detections']}")
    print(f"Traffic categories detected: {stats['traffic_categories_detected']}/{stats['supported_category_count']}")
    print(f"Traffic/object counts: {stats['traffic_counts']}")
    print(f"CSV report: {stats['report_path']}")
    print(f"Violation screenshots: {stats['violations_dir']}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live helmet violation tracking with ByteTrack")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to YOLO model weights")
    parser.add_argument(
        "--traffic-model",
        default=str(DEFAULT_TRAFFIC_MODEL_PATH),
        help="Path to COCO traffic/object YOLO model. Use yolo11s.pt for the final YOLOv11s demo.",
    )
    args = parser.parse_args()

    detect_webcam(
        confidence=args.confidence,
        camera_index=args.camera,
        model_path=args.model,
        traffic_model_path=args.traffic_model,
    )
