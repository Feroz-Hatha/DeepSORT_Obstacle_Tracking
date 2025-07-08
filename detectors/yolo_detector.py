import torch
import numpy as np

class YOLODetector:
    def __init__(self, model_name='yolov5s', device='cpu', conf_threshold=0.3):
        """
        Initializes YOLOv5 detector.

        :param model_name: Model variant to use (e.g., 'yolov5s', 'yolov5m', etc.)
        :param device: 'cpu' or 'cuda'
        :param conf_threshold: Minimum confidence threshold for detections
        """
        self.device = device
        self.conf_threshold = conf_threshold

        # Load model from Ultralytics repo
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame):
        """
        Detect objects in a single frame.

        :param frame: Input frame (numpy array BGR)
        :return: List of detections: [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        # YOLO expects RGB images
        img_rgb = frame[..., ::-1]

        results = self.model(img_rgb)

        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box
            detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        return detections
