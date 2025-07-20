<h1 align="center">🚦 Density Based AI-Driven Intelligent Traffic Control System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-Computer_Vision-orange?style=for-the-badge&logo=ultralytics" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/AI-Traffic_Control-brightgreen?style=for-the-badge" alt="AI Traffic Control">
</p>

<p align="center"><strong>Revolutionizing Urban Traffic Management through AI-Powered Real-Time Adaptive Signal Control</strong></p>

<p align="center">
📖 <b>Read Full Thesis</b> • 🎥 <b>Demo Videos</b> • 🚀 <b>Quick Start</b> • 📊 <b>Performance</b>
</p>

---

## 🌟 Overview

The **Density Based AI Driven Intelligent Traffic Control System** leverages YOLOv8 and real-time algorithms to intelligently manage urban intersections. Developed at **Chittagong University of Engineering and Technology**, this solution addresses traffic congestion, improves emergency response, and supports mixed traffic conditions.

---

## 🎯 Key Achievements

* ✅ **82.09%** F1-Score for vehicle detection
* ⏱️ **42%** reduction in average vehicle delays
* 🚑 **26%** faster emergency vehicle response time
* 🌍 Real-world tested in **Dhaka traffic conditions**

---

## 🚀 Features

### 🔥 Core Capabilities

* 🤖 **AI-Powered Detection**: Custom-trained YOLOv8 detecting 12 vehicle types
* ⚡ **Real-Time Adaptive Signals**: Dynamically adjusts based on traffic density
* 🚨 **Emergency Vehicle Priority**: Detects and routes ambulances & fire trucks
* 📊 **Tracking System**: Kalman Filter + Hungarian Algorithm
* 🌟 **High Performance**: 15 FPS on edge devices
* 💸 **Cost-Effective**: Works with existing CCTV infrastructure

---

## 🚗 Vehicle Detection Classes

The model detects 12 common vehicle types in Bangladesh:

```
🚴 bicycle     🏍️ bike      🚌 bus       🚗 car  
🙺 cng         🛵 easybike  🚐 leguna    🛼 rickshaw  
🚚 truck       🚙 van       🚑 ambulance 🚒 fire truck
```

---

## 📊 Performance Metrics

### 🎯 Model Comparison

| Metric        | YOLOv8m (Ours) | Faster R-CNN | SSD    |
| ------------- | -------------- | ------------ | ------ |
| Precision     | ✅ 86.49%       | 78.00%       | 81.50% |
| Recall        | ✅ 78.12%       | 74.00%       | 77.00% |
| mAP\@0.5      | ✅ 83.79%       | 76.00%       | 79.00% |
| F1-Score      | ✅ 82.09%       | 75.86%       | 79.22% |
| Inference FPS | ⚡ 15 FPS       | 3 FPS        | 10 FPS |

---

## 🚦 Traffic Flow Improvements

* ⏳ **Vehicle Delays**: 42% reduction vs fixed-time signals
* 🚑 **Emergency Response**: 26% improvement
* ✅ **System Accuracy**: 95–100% in real-world test
* ↺ **Latency**: <30s for full detection-to-signal cycle

---

## 🎥 Demo Videos (Coming Soon)

* 🔍 **Vehicle Detection**: Real-time classification
* 🚦 **Traditional vs AI Control**:

  * **Normal Mode**: Fixed-time signals
  * **Smart Mode**: Adaptive signal system

---

## 🚀 Quick Start

### 📋 Prerequisites

* Python 3.8+
* CUDA-capable GPU (recommended)
* OpenCV 4.0+
* 8 GB+ RAM

### ⚙️ Installation

```bash
git clone https://github.com/yourusername/density-based-ai-traffic-control.git
cd density-based-ai-traffic-control
pip install -r requirements.txt
```

Place your trained model in the root:

```bash
# Example
ayman.pt
```

Update config in the Python script:

```python
MODEL_PATH = 'ayman.pt'
VIDEO_PATH = 'your_input_video.mp4'
OUTPUT_PATH = 'output_video.mp4'
CONFIDENCE_THRESHOLD = 0.3
```

### 🎮 Run Detection

```bash
python traffic_detection.py
```

✅ **What it does**:

* Loads YOLOv8 model
* Detects & tracks vehicles
* Analyzes density
* Outputs annotated video
* Shows real-time stats

---

## 📊 Output Features

* 🔢 Per-lane vehicle counting
* 🚨 Emergency vehicle alerts
* 📈 Density & traffic stats
* 🎥 Visual overlays (boxes, labels, stats)

---

## 🏗️ System Architecture

```text
📹 CCTV Camera Feed
       ↓
🔍 YOLOv8 Vehicle Detection
       ↓
📊 Multi-Object Tracking
       ↓
📈 Density Analysis
       ↓
🧠 Adaptive Signal Control
       ↓
🚦 Traffic Signal Output
```

---

## 🛠️ Core Components

* 🎥 Video Input Module
* 🧠 YOLOv8 Detection Engine
* 📍 Vehicle Tracking
* 📊 Traffic Density Calculator
* 🚦 Adaptive Signal Controller
* 🚑 Emergency Detection Logic

---

## 🤖 AI Model Specs

* Model: YOLOv8m
* Dataset: 6,366 Dhaka traffic images
* Classes: 12
* Platform: Kaggle (Tesla T4)
* Training: 150 epochs, dynamic LR

---

## 🧠 Algorithm Details

* 📍 Tracking: Kalman Filter + Hungarian Algorithm
* 📈 Smoothing: Weighted moving average
* 🚦 Signal Logic: Proportional to density
* 🚑 Emergency Detection: Multi-frame confirmation

---

## 🎓 Research Contribution

* 🏩 **Urban Congestion**: Real-time adaptive solution
* 🚑 **Emergency Routing**: Life-saving detection
* 🌍 **Bangladesh Focused**: Local traffic classes
* 💰 **Low-Cost**: Uses existing infrastructure

---

## 📁 Publication Details

* 📘 Thesis: *YOLOv8-based Intelligent Traffic Control System*
* 🏫 Institution: CUET
* 👨‍🏫 Supervisor: S.M. Fahim Faisal
* 🗓️ Year: 2025

---

## 🤝 Contributing

### ⚙️ Setup

```bash
# Fork the repo
# Create a branch
git checkout -b feature/AmazingFeature
# Make changes and commit
git commit -m "Add AmazingFeature"
# Push and create PR
git push origin feature/AmazingFeature
```

### 📋 Contribution Ideas

* 🎯 Improve detection accuracy
* ⚡ Optimize FPS
* 🌙 Support night/weather visibility
* 📱 Build mobile version
* 🌍 Add more regional vehicles

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍🎓 Author

**Ayman Ilias**<br>
🎓 Student ID: 1909025<br>
🏫 CUET - Mechatronics and Industrial Engineering<br>
📧 Email: [aymanilias00@gmail.com)
💼 [LinkedIn]([https://www.linkedin.com/i](https://www.linkedin.com/in/aymanilias/))<br>
🐙 [GitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

Special thanks to:

* 🎓 S.M. Fahim Faisal (Supervisor) - > Assistant Professor,MIE,CUET

---

## ⭐ Final Note

<p align="center">
  ⭐ If this project helps you, give it a star! ⭐ <br><br>
  🚦 Building smarter cities, one intersection at a time 🏛️
</p>
