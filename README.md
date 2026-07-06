# AI Traffic Safety & Helmet Violation Monitoring System

This project combines a custom helmet detector with a COCO-pretrained YOLO traffic/object detector. It detects 15+ object categories, tracks objects in video/webcam streams, counts helmet violations once per tracking ID, and stores violation evidence.

## Main Features

- Image helmet detection
- Video helmet detection
- Live webcam detection
- Multi-class traffic/object detection with 15+ categories
- ByteTrack object tracking
- One-time violation counting per tracking ID
- Sound alert for each new violator
- Screenshot evidence saved for each new violation
- CSV violation report generation
- FPS, violation, safe detection, and duplicate-reduction statistics
- Tkinter GUI for image upload, video processing, webcam demo, statistics, and saved evidence

## Dataset

The included dataset is in `../Helmet_Detection_DataSet` and contains:

| Split | Images |
| --- | ---: |
| Train | 7308 |
| Validation | 462 |
| Test | 254 |

Custom helmet dataset classes:

- `with helmet`
- `without helmet`

The custom dataset satisfies the 500+ sample requirement. The 15+ category requirement is handled by the COCO-pretrained YOLO traffic model.

## 15+ Detectable Categories

The system detects helmet classes plus COCO traffic/environment classes:

```text
helmet
no_helmet
person
bicycle
car
motorcycle
bus
truck
traffic light
stop sign
backpack
umbrella
handbag
cell phone
dog
cat
chair
bench
fire hydrant
parking meter
skateboard
```

That gives 21 supported categories, exceeding the requirement.

## Install

```bash
pip install -r requirements.txt
```

`pygame` is used for the alert sound. If `alert.wav` is present in this folder, it will play when a new violation ID is counted. On Windows, the system falls back to a short beep if `pygame` or `alert.wav` is unavailable.

## Run

Image detection:

```bash
python detect_single_image.py pic_test.jpg --confidence 0.5
```

Video tracking:

```bash
python detect_video.py bike_2.mp4 --confidence 0.5
```

The default traffic/object detector is now `yolo11s.pt`, so the final demo uses YOLOv11s for COCO multi-class detection:

```bash
python detect_video.py bike_2.mp4
```

For faster low-resource testing, you can use the smaller local fallback:

```bash
python detect_video.py bike_2.mp4 --traffic-model Model/yolo11n.pt
```

Webcam tracking:

```bash
python detect_webcam.py --camera 0 --confidence 0.5
```

Tkinter GUI:

```bash
python gui.py
```

## Outputs

Violation screenshots:

```text
output/violations/
```

CSV report:

```text
output/violation_report.csv
```

Tracked videos:

```text
output/video_results/
```

CSV columns:

```text
violation_id,time,source,track_id,class_name,confidence,image_path
```

## Evaluation Metrics

Run model evaluation on the test set:

```bash
python evaluate_metrics.py --split test
```

The metrics report is saved to:

```text
output/evaluation_metrics_test.csv
```

## YOLOv11s Training

The tracking and reporting system works with any Ultralytics-compatible helmet model. To train a YOLOv11s model on the included dataset:

```bash
python train_yolo11s.py --epochs 50 --imgsz 640 --batch 8
```

After training, use the generated `runs/train/helmet_yolo11s/weights/best.pt` as the model path in the scripts or dashboard.

Report these values in the final demo:

- Precision
- Recall
- mAP50
- mAP50-95
- FPS
- Total unique violations
- Duplicate count reduction

## Architecture

```text
Input: Image / Video / Webcam
        |
Custom Helmet YOLO + COCO YOLO Traffic Detection
        |
Multi-Class Traffic Detection
        |
ByteTrack Object Tracking
        |
Helmet / No-Helmet Classification
        |
If no helmet and new tracking ID:
        |
Count Violation + Play Alert + Save Screenshot + Write CSV
        |
Display Boxes + Labels + Confidence + FPS + Count
```

## Improvement Summary

| Feature | Old System | Improved System |
| --- | --- | --- |
| Detection | Yes | Yes |
| Object Categories | 2 helmet classes | 21 helmet + traffic/object classes |
| Video Counting | Repeated detections | Count once per tracking ID |
| Webcam | Yes | Yes |
| Sound Alert | No | Yes |
| Tracking ID | No | ByteTrack |
| Report CSV | No | Yes |
| Screenshot Evidence | No | Yes |
| FPS Display | Basic | Yes |
