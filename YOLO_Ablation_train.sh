#!/bin/bash

cd /home/ubuntu/your_path

# 全增强
yolo detect train model=yolov8m.pt data=data/yolov5pytorch_PI/data.yaml epochs=350 batch=16 imgsz=640 device=0 mosaic=1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 flipud=0.5 fliplr=0.5 name=exp_full

# No mosaic
yolo detect train model=yolov8m.pt data=data/yolov5pytorch_PI/data.yaml epochs=350 batch=16 imgsz=640 device=0 mosaic=0 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 flipud=0.5 fliplr=0.5 name=exp_no_mosaic

# No hsv
yolo detect train model=yolov8m.pt data=data/yolov5pytorch_PI/data.yaml epochs=350 batch=16 imgsz=640 device=0 mosaic=1 hsv_h=0 hsv_s=0 hsv_v=0 flipud=0.5 fliplr=0.5 name=exp_no_hsv

# No flip
yolo detect train model=yolov8m.pt data=data/yolov5pytorch_PI/data.yaml epochs=350 batch=16 imgsz=640 device=0 mosaic=1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 flipud=0 fliplr=0 name=exp_no_flip

# No pretrain
yolo detect train model=yolov8m.yaml data=data/yolov5pytorch_PI/data.yaml epochs=350 batch=16 imgsz=640 device=0 mosaic=1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 flipud=0.5 fliplr=0.5 name=exp_no_pretrain

# Tiny backbone
yolo detect train model=yolov8n.pt data=data/yolov5pytorch_PI/data.yaml epochs=350 batch=16 imgsz=640 device=0 mosaic=1 hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 flipud=0.5 fliplr=0.5 name=exp_tiny_backbone
