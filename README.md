# Deep-Learning-Capstone-Project
Real-time safety helmet compliance monitoring system using an optimized lightweight YOLO detector and ONNX Runtime with automated visual and debounced auditory non-compliance alerts.

# Real-Time Safety Helmet Detection and Compliance Monitoring System

An automated, edge-optimized computer vision pipeline designed to monitor PPE compliance at entry gates, workshops, construction sites, and restricted industrial environments in real time[cite: 1]. 

The system detects individuals, verifies whether they are wearing safety helmets, annotates visual compliance status, and triggers both on-screen visual warnings and auditory alarms upon detecting safety violations[cite: 1].

---

## 🚀 Key Features

* **Real-Time Detection:** Evaluates live video feeds from standard webcams, RTSP IP camera streams, or recorded video files using OpenCV with low-latency capture backends[cite: 1].
* **Dual-Class Bounding Box Classification:** Distinguishes between compliant heads (`helmet`) and non-compliant individuals (`no_helmet`)[cite: 1].
* **Edge-Optimized Inference:** Exported to the Open Neural Network Exchange (ONNX) format, enabling high-FPS, lightweight CPU execution via ONNX Runtime without requiring a dedicated high-end GPU[cite: 1].
* **Debounced Auditory & Visual Alerting:**
  * **Status Banners:** Color-coded indicators (`COMPLIANT`, `WARNING: NO HELMET - ACCESS DENIED`, and `NO PERSON DETECTED`)[cite: 1].
  * **False-Alarm Suppression:** Debounce logic triggers an alarm only after a streak of consecutive violating frames (e.g., 3 frames) with a built-in alert cooldown to eliminate transient detection glitches[cite: 1].
* **Robust Domain Generalization:** Trained with extensive augmentations—including HSV photometric perturbations, scaling, perspective shifts, and mosaic mixing—to withstand uneven industrial lighting and distance variations[cite: 1].

---

## 🛠️ Tech Stack & Architecture

* **Frameworks & Libraries:** Python, PyTorch, Ultralytics YOLOv8, ONNX / ONNX Runtime, OpenCV[cite: 1]
* **Dataset:** Safety Helmet Wearing Dataset (SHWD) (~7,500+ annotated images converted from Pascal VOC XML to normalized YOLO format)[cite: 1]
* **Model:** Lightweight Nano Detector (`YOLOv8n`)[cite: 1]
* **Deployment Backends:** DirectShow (Windows) / V4L2 (Linux) for zero-latency camera I/O[cite: 1]

---

## 📊 Performance & Evaluation

* **Model Footprint:** ~6 MB (`.pt`) / ~12 MB (`.onnx`)[cite: 1]
* **Inference Speed:** Real-time throughput (15–30+ FPS on consumer CPUs; 100+ FPS on NVIDIA GPUs)[cite: 1]
* **Key Metrics:** Precision, Recall, F1-Score, mAP@0.5, and mAP@0.5:0.95 evaluated across varying distance and lighting conditions[cite: 1].

---

## 💻 Quick Start

### 1. Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/safety-helmet-detection.git
cd safety-helmet-detection
pip install opencv-python ultralytics onnxruntime
