import cv2
import numpy as np
import os
import warnings
from detectors.yolo_detector import YOLODetector
from trackers.sort import Sort
from deep_feature_extractor import FeatureExtractor

warnings.simplefilter(action='ignore', category=FutureWarning)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def draw_boxes(frame, tracks, coco_classes):
    for d in tracks:
        x1, y1, x2, y2, track_id, class_id = d
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (int(track_id * 37) % 255, int(track_id * 17) % 255, int(track_id * 29) % 255)

        label = f"ID {int(track_id)} {coco_classes[int(class_id)]}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    return frame

def main():
    os.makedirs("output", exist_ok=True)

    # Initialize detector and feature extractor
    detector = YOLODetector(model_name='yolov5s', device='cpu', conf_threshold=0.3)
    extractor = FeatureExtractor(device='cpu')

    # Initialize Deep SORT tracker
    tracker = Sort()

    input_path = "data/KITTI/training/image_02/0002.mp4"
    output_path = "output/kitti_0002_DeepSORT.mp4"

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame)

        # Prepare detection array and features
        if detections:
            dets_np = np.array([[*d[:4], d[4], d[5]] for d in detections])
            features = []

            for d in detections:
                x1, y1, x2, y2 = [int(coord) for coord in d[:4]]
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    # If the crop is invalid (outside image), use a zero vector
                    features.append(np.zeros(512))
                else:
                    feat = extractor.extract(crop)
                    features.append(feat)
        else:
            dets_np = np.empty((0, 6))
            features = []

        # Update tracker with detections and features
        tracks = tracker.update(dets_np, features)

        # Draw results
        frame = draw_boxes(frame, tracks, COCO_CLASSES)

        # Write and show
        out.write(frame)
        cv2.imshow("Deep SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing completed! Output saved to: {output_path}")

if __name__ == "__main__":
    main()