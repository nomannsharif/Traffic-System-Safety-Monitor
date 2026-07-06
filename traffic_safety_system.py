from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import csv
import os
import threading
import time

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "best.pt"
DEFAULT_TRAFFIC_MODEL_PATH = PROJECT_DIR / "yolo11s.pt"
FAST_TRAFFIC_MODEL_PATH = PROJECT_DIR / "Model" / "yolo11n.pt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_VIOLATIONS_DIR = DEFAULT_OUTPUT_DIR / "violations"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "violation_report.csv"

TRAFFIC_OBJECT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "backpack",
    "umbrella",
    "handbag",
    "cell phone",
    "dog",
    "cat",
    "chair",
    "bench",
    "fire hydrant",
    "parking meter",
    "skateboard",
}

HELMET_OBJECT_CLASSES = {"helmet", "no_helmet"}


def is_violation_label(label):
    normalized = label.lower().replace("_", " ").replace("-", " ")
    return "without" in normalized or "no helmet" in normalized or "nohelmet" in normalized


def is_safe_label(label):
    normalized = label.lower().replace("_", " ").replace("-", " ")
    return "with helmet" in normalized or "helmet" in normalized


@dataclass
class DetectionRecord:
    track_id: str
    class_name: str
    confidence: float
    bbox: tuple
    is_violation: bool
    is_new_violation: bool = False
    source_model: str = "helmet"
    is_traffic_object: bool = False


class SoundAlert:
    def __init__(self, sound_path=None, enabled=True):
        self.enabled = enabled
        self.sound = None
        self.pygame = None
        self.sound_path = Path(sound_path) if sound_path else PROJECT_DIR / "alert.wav"

        if not enabled:
            return

        try:
            import pygame

            pygame.mixer.init()
            self.pygame = pygame
            if self.sound_path.exists():
                self.sound = pygame.mixer.Sound(str(self.sound_path))
        except Exception:
            self.pygame = None
            self.sound = None

    def play(self):
        if not self.enabled:
            return

        def _play():
            try:
                if self.sound is not None:
                    self.sound.play()
                    return

                if os.name == "nt":
                    import winsound

                    winsound.Beep(1200, 180)
            except Exception:
                pass

        threading.Thread(target=_play, daemon=True).start()


class ViolationLogger:
    fieldnames = [
        "violation_id",
        "time",
        "source",
        "track_id",
        "class_name",
        "confidence",
        "image_path",
    ]

    def __init__(
        self,
        report_path=DEFAULT_REPORT_PATH,
        violations_dir=DEFAULT_VIOLATIONS_DIR,
        alert=None,
    ):
        self.report_path = Path(report_path)
        self.violations_dir = Path(violations_dir)
        self.alert = alert or SoundAlert(enabled=False)
        self.violated_ids = set()
        self.next_violation_id = 1
        self.raw_violation_detections = 0
        self.safe_detections = 0
        self.violations_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_report()

    def _ensure_report(self):
        if self.report_path.exists() and self.report_path.stat().st_size > 0:
            try:
                with self.report_path.open("r", newline="", encoding="utf-8") as csv_file:
                    rows = list(csv.DictReader(csv_file))
                existing_ids = [int(row["violation_id"]) for row in rows if row.get("violation_id")]
                if existing_ids:
                    self.next_violation_id = max(existing_ids) + 1
            except Exception:
                pass
            return

        with self.report_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()

    def reset_session(self):
        self.violated_ids.clear()
        self.raw_violation_detections = 0
        self.safe_detections = 0

    @property
    def unique_violations(self):
        return len(self.violated_ids)

    @property
    def duplicate_reduction(self):
        return max(0, self.raw_violation_detections - self.unique_violations)

    def log_if_new(self, frame, source, track_id, class_name, confidence, bbox):
        self.raw_violation_detections += 1
        track_key = str(track_id)
        if track_key in self.violated_ids:
            return False, None

        self.violated_ids.add(track_key)

        violation_id = self.next_violation_id
        self.next_violation_id += 1

        now = datetime.now()
        timestamp = now.strftime("%H-%M-%S")
        image_name = f"violation_{violation_id:03d}_{timestamp}_id-{track_key}.jpg"
        image_path = self.violations_dir / image_name

        evidence = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            evidence,
            f"Violation #{violation_id} ID {track_key} {confidence:.2f}",
            (max(0, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(str(image_path), evidence)

        with self.report_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow(
                {
                    "violation_id": violation_id,
                    "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": source,
                    "track_id": track_key,
                    "class_name": class_name,
                    "confidence": f"{confidence:.4f}",
                    "image_path": str(image_path),
                }
            )

        self.alert.play()
        return True, str(image_path)


class HelmetViolationSystem:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        traffic_model_path=DEFAULT_TRAFFIC_MODEL_PATH,
        confidence=0.5,
        traffic_confidence=0.35,
        enable_traffic_detection=True,
        report_path=DEFAULT_REPORT_PATH,
        violations_dir=DEFAULT_VIOLATIONS_DIR,
        alert_sound_path=None,
        enable_sound=True,
    ):
        self.model_path = Path(model_path)
        self.traffic_model_path = Path(traffic_model_path) if traffic_model_path else None
        self.confidence = confidence
        self.traffic_confidence = traffic_confidence
        self.enable_traffic_detection = enable_traffic_detection
        self.model = YOLO(str(self.model_path))
        self.traffic_model = None
        if enable_traffic_detection and self.traffic_model_path:
            self.traffic_model = YOLO(str(self.traffic_model_path))
        self.alert = SoundAlert(alert_sound_path, enabled=enable_sound)
        self.logger = ViolationLogger(report_path, violations_dir, alert=self.alert)
        self.traffic_track_ids = {class_name: set() for class_name in TRAFFIC_OBJECT_CLASSES}
        self.frame_count = 0
        self.fps = 0
        self._fps_frames = 0
        self._fps_started_at = time.time()

    def reset_session(self):
        self.logger.reset_session()
        self.traffic_track_ids = {class_name: set() for class_name in TRAFFIC_OBJECT_CLASSES}
        self.frame_count = 0
        self.fps = 0
        self._fps_frames = 0
        self._fps_started_at = time.time()

    def stats(self):
        traffic_counts = {
            class_name: len(ids)
            for class_name, ids in sorted(self.traffic_track_ids.items())
            if ids
        }
        return {
            "frames": self.frame_count,
            "fps": self.fps,
            "unique_violations": self.logger.unique_violations,
            "raw_violation_detections": self.logger.raw_violation_detections,
            "duplicate_count_reduction": self.logger.duplicate_reduction,
            "safe_detections": self.logger.safe_detections,
            "traffic_counts": traffic_counts,
            "traffic_categories_detected": len(traffic_counts),
            "total_unique_traffic_objects": sum(traffic_counts.values()),
            "supported_categories": sorted(TRAFFIC_OBJECT_CLASSES | HELMET_OBJECT_CLASSES),
            "supported_category_count": len(TRAFFIC_OBJECT_CLASSES | HELMET_OBJECT_CLASSES),
            "report_path": str(self.logger.report_path),
            "violations_dir": str(self.logger.violations_dir),
        }

    def process_frame(self, frame, source="video", confidence=None, imgsz=640, persist=True):
        confidence = self.confidence if confidence is None else confidence
        self.frame_count += 1
        self._fps_frames += 1

        elapsed = time.time() - self._fps_started_at
        if elapsed >= 1.0:
            self.fps = int(self._fps_frames / elapsed)
            self._fps_frames = 0
            self._fps_started_at = time.time()

        detections = []
        annotated = frame.copy()

        if self.traffic_model is not None:
            traffic_result = self.traffic_model.track(
                frame,
                conf=self.traffic_confidence,
                imgsz=imgsz,
                persist=persist,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
            detections.extend(self._collect_traffic_detections(traffic_result))

        result = self.model.track(
            frame,
            conf=confidence,
            imgsz=imgsz,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        for index, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = self.model.names[cls]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            track_id = "untracked"
            if box.id is not None:
                track_id = str(int(box.id[0]))
            else:
                track_id = f"frame-{self.frame_count}-det-{index}"

            violation = is_violation_label(class_name)
            new_violation = False

            if violation:
                new_violation, _ = self.logger.log_if_new(
                    frame=frame,
                    source=source,
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf_score,
                    bbox=(x1, y1, x2, y2),
                )
            else:
                self.logger.safe_detections += 1

            detections.append(
                DetectionRecord(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf_score,
                    bbox=(x1, y1, x2, y2),
                    is_violation=violation,
                    is_new_violation=new_violation,
                    source_model="helmet",
                    is_traffic_object=False,
                )
            )

        self._draw_overlay(annotated, detections)
        return annotated, detections, self.stats()

    def process_image(self, image_path, confidence=None, imgsz=640, source="image"):
        confidence = self.confidence if confidence is None else confidence
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        result = self.model.predict(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
        annotated = image.copy()
        detections = []

        if self.traffic_model is not None:
            traffic_result = self.traffic_model.predict(
                image,
                conf=self.traffic_confidence,
                imgsz=imgsz,
                verbose=False,
            )[0]
            detections.extend(
                self._collect_traffic_detections(traffic_result, image_key=Path(image_path).stem)
            )

        for index, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = self.model.names[cls]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            track_id = f"image-{Path(image_path).stem}-{index + 1}"
            violation = is_violation_label(class_name)
            new_violation = False

            if violation:
                new_violation, _ = self.logger.log_if_new(
                    frame=image,
                    source=source,
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf_score,
                    bbox=(x1, y1, x2, y2),
                )
            else:
                self.logger.safe_detections += 1

            detections.append(
                DetectionRecord(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf_score,
                    bbox=(x1, y1, x2, y2),
                    is_violation=violation,
                    is_new_violation=new_violation,
                    source_model="helmet",
                    is_traffic_object=False,
                )
            )

        self._draw_overlay(annotated, detections)
        self.frame_count += 1
        return annotated, detections, self.stats()

    def process_video(
        self,
        video_path,
        output_path=None,
        confidence=None,
        imgsz=640,
        show_preview=False,
    ):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            output_dir = DEFAULT_OUTPUT_DIR / "video_results"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"tracked_{Path(video_path).name}"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed += 1
            annotated, _, stats = self.process_frame(
                frame,
                source=f"video:{Path(video_path).name}",
                confidence=confidence,
                imgsz=imgsz,
                persist=True,
            )
            self._draw_status(annotated, stats, processed, total_frames)
            writer.write(annotated)

            if show_preview:
                cv2.imshow("Smart Helmet Violation Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        writer.release()
        if show_preview:
            self._safe_destroy_windows()
        return str(output_path), self.stats()

    def process_webcam(self, camera_index=0, confidence=None, imgsz=640):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError("Cannot access webcam")

        self.reset_session()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _, stats = self.process_frame(
                frame,
                source=f"webcam:{camera_index}",
                confidence=confidence,
                imgsz=imgsz,
                persist=True,
            )
            self._draw_status(annotated, stats)
            cv2.imshow("Smart Helmet Violation Detection - Webcam", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        self._safe_destroy_windows()
        return self.stats()

    def _draw_overlay(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            if detection.is_traffic_object:
                color = (255, 140, 0)
                label = f"{detection.class_name.upper()} ID {detection.track_id} {detection.confidence:.2f}"
            elif detection.is_violation:
                color = (0, 0, 255)
                label = f"NO HELMET ID {detection.track_id} {detection.confidence:.2f}"
            else:
                color = (0, 180, 0)
                label = f"HELMET ID {detection.track_id} {detection.confidence:.2f}"

            if detection.is_new_violation:
                label = f"NEW {label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(0, y1 - text_h - 10)
            cv2.rectangle(frame, (x1, y_text), (x1 + text_w + 8, y_text + text_h + 8), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y_text + text_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

    def _collect_traffic_detections(self, result, image_key=None):
        detections = []
        for index, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            class_name = self.traffic_model.names[cls]
            if class_name not in TRAFFIC_OBJECT_CLASSES:
                continue

            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            if box.id is not None:
                track_id = str(int(box.id[0]))
            elif image_key:
                track_id = f"image-{image_key}-{class_name}-{index + 1}"
            else:
                track_id = f"frame-{self.frame_count}-{class_name}-{index + 1}"

            self.traffic_track_ids.setdefault(class_name, set()).add(str(track_id))
            detections.append(
                DetectionRecord(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf_score,
                    bbox=(x1, y1, x2, y2),
                    is_violation=False,
                    is_new_violation=False,
                    source_model="coco",
                    is_traffic_object=True,
                )
            )

        return detections

    def _draw_status(self, frame, stats, frame_index=None, total_frames=None):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        lines = [
            f"FPS: {stats['fps']}",
            f"Unique violations: {stats['unique_violations']}",
            f"Duplicate reduction: {stats['duplicate_count_reduction']}",
            f"Traffic categories: {stats['traffic_categories_detected']}/{stats['supported_category_count']}",
        ]
        if frame_index is not None and total_frames:
            lines.append(f"Frame: {frame_index}/{total_frames}")

        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, 38 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _safe_destroy_windows(self):
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="AI Traffic Safety & Helmet Violation Monitoring System"
    )
    parser.add_argument(
        "--mode",
        choices=["image", "video", "webcam"],
        default="image",
        help="Run image, video, or webcam detection",
    )
    parser.add_argument(
        "--source",
        default=str(PROJECT_DIR / "pic_test.jpg"),
        help="Image/video path. Ignored for webcam mode.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index for webcam mode",
    )
    parser.add_argument(
        "--helmet-model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to custom helmet/no-helmet model",
    )
    parser.add_argument(
        "--traffic-model",
        default=str(DEFAULT_TRAFFIC_MODEL_PATH),
        help="Path to COCO traffic/object model",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Helmet model confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO image size",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV preview for video mode",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable alert sound",
    )
    args = parser.parse_args()

    system = HelmetViolationSystem(
        model_path=args.helmet_model,
        traffic_model_path=args.traffic_model,
        confidence=args.confidence,
        enable_sound=not args.no_sound,
    )

    if args.mode == "image":
        annotated, detections, stats = system.process_image(
            args.source,
            confidence=args.confidence,
            imgsz=args.imgsz,
            source="cli:image",
        )
        output_dir = DEFAULT_OUTPUT_DIR / "image_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detected_{Path(args.source).name}"
        cv2.imwrite(str(output_path), annotated)

        print(f"Image processed: {args.source}")
        print(f"Output image: {output_path}")
        print(f"Detections: {len(detections)}")

    elif args.mode == "video":
        output_path, stats = system.process_video(
            args.source,
            confidence=args.confidence,
            imgsz=args.imgsz,
            show_preview=args.preview,
        )
        print(f"Video processed: {args.source}")
        print(f"Output video: {output_path}")

    else:
        stats = system.process_webcam(
            camera_index=args.camera,
            confidence=args.confidence,
            imgsz=args.imgsz,
        )
        print("Webcam session ended")

    print(f"Supported categories: {stats['supported_category_count']}")
    print(f"Traffic categories detected: {stats['traffic_categories_detected']}")
    print(f"Traffic/object counts: {stats['traffic_counts']}")
    print(f"Unique violations: {stats['unique_violations']}")
    print(f"Raw violation detections: {stats['raw_violation_detections']}")
    print(f"Duplicate count reduction: {stats['duplicate_count_reduction']}")
    print(f"CSV report: {stats['report_path']}")
    print(f"Violation screenshots: {stats['violations_dir']}")


if __name__ == "__main__":
    main()
