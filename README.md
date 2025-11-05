# 🧠 Footfall Counter using Computer Vision

This project automatically counts the number of people entering and exiting a region (such as a doorway or store entrance) using **YOLOv8 object detection** and **OpenCV**.

---

## 🎯 Overview

The system:
- Detects people in video frames using **YOLOv8 (pretrained on COCO dataset)**  
- Tracks their movement across a **counting line**
- Determines whether a person **entered** or **exited** based on direction
- Displays and saves results with bounding boxes and counts

---

## 🧩 Features

✅ Real-time person detection with YOLOv8  
✅ Centroid-based tracking for direction detection  
✅ Dynamic counting line (horizontal or vertical)  
✅ Adjustable confidence threshold for better results  
✅ Output video saved with visual annotations  

---
