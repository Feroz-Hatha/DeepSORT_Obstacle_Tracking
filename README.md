# Real-Time Multi-Object Tracking (SORT & Deep SORT)

This project implements real-time multi-object tracking using SORT and Deep SORT, designed for dynamic scenes with vehicles and pedestrians (e.g., autonomous driving scenarios).

## Features
- Kalman filter-based motion prediction (SORT)
- Appearance-based re-identification embeddings (Deep SORT with ResNet features)
- Hybrid cost matrix combining IoU and appearance similarity
- COCO object class overlay with track IDs
- Modular PyTorch + OpenCV pipeline
- KITTI benchmark video integration

## Usage
```bash
python main.py