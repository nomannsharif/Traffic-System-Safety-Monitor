import argparse
from pathlib import Path

from helmet_violation_system import HelmetViolationSystem, DEFAULT_MODEL_PATH, DEFAULT_TRAFFIC_MODEL_PATH


def detect_video(
    video_path,
    confidence=0.5,
    show_preview=False,
    model_path=DEFAULT_MODEL_PATH,
    traffic_model_path=DEFAULT_TRAFFIC_MODEL_PATH,
):
    system = HelmetViolationSystem(
        model_path=model_path,
        traffic_model_path=traffic_model_path,
        confidence=confidence,
    )

    print(f"Processing video with ByteTrack: {video_path}")
    output_path, stats = system.process_video(
        video_path=video_path,
        confidence=confidence,
        show_preview=show_preview,
    )

    print("Video processing complete")
    print(f"Output video: {output_path}")
    print(f"Unique violations: {stats['unique_violations']}")
    print(f"Raw violation detections: {stats['raw_violation_detections']}")
    print(f"Duplicate count reduction: {stats['duplicate_count_reduction']}")
    print(f"Safe detections: {stats['safe_detections']}")
    print(f"Traffic categories detected: {stats['traffic_categories_detected']}/{stats['supported_category_count']}")
    print(f"Traffic/object counts: {stats['traffic_counts']}")
    print(f"CSV report: {stats['report_path']}")
    print(f"Violation screenshots: {stats['violations_dir']}")
    return output_path, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Helmet Violation Detection & Tracking")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "bike_2.mp4"),
        help="Path to the input video",
    )
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV preview while processing")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to YOLO model weights")
    parser.add_argument(
        "--traffic-model",
        default=str(DEFAULT_TRAFFIC_MODEL_PATH),
        help="Path to COCO traffic/object YOLO model. Use yolo11s.pt for the final YOLOv11s demo.",
    )
    args = parser.parse_args()

    detect_video(
        video_path=args.video_path,
        confidence=args.confidence,
        show_preview=args.preview,
        model_path=args.model,
        traffic_model_path=args.traffic_model,
    )
