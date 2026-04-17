# 🎯 Offline Multi-Object Tracking (MOT) with Missing Frame Prediction
## 🚀 ML Code Crafters - Project Overview


### 📹 Project Summary
This repository contains a high-performance **Offline Multi-Object Tracking (MOT)** system designed for UAV (Unmanned Aerial Vehicle) surveillance videos. Leveraging the **VisDrone2019-MOT** dataset, our system focuses on robust trajectory reconstruction and tracklet association using a combination of classical computer vision and machine learning techniques.

The core innovation lies in our **Hybrid Trajectory Reconstruction Engine**, which simulates real-world challenges like frame drops and occlusions, and recovers lost object paths with high precision.

---

### ✨ Key Features
- **🛠️ Robust Offline Pipeline**: Specifically optimized for post-processing and research-quality trajectory analysis.
- **📉 Missing Frame Simulation**: Dynamic simulation of random and continuous frame drops to test system resilience.
- **🧠 Hybrid Prediction Engine**:
  - **Kalman Filter**: Constant velocity model for temporal smoothed prediction.
  - **Random Forest Regressor**: ML-driven movement estimation for complex trajectories.
  - **Linear Interpolation**: Reliable baseline for short-term gap filling.
- **🔗 Intelligent Tracklet Merging**: ML-based classification for long-term object association across wide temporal gaps.
- **📊 Comprehensive Benchmarking**: Evaluation using standard MOT metrics: **MOTA, IDF1, HOTA, MOTP**, and **IoU**.

---

### 📂 Project Structure

| Directory/File | Description |
| :--- | :--- |
| `offline_mot_tracking.ipynb` | Main notebook implementing the tracking pipeline and evaluation. |
| `VisDrone2019-MOT-train/` | Root directory for the VisDrone dataset (Images & Annotations). |
| `tracking_output/` | Generated tracking results and visualizations. |
| `reports/` | Mid-sem and End-sem project reports and presentations. |

---

### 🛠️ Methodology

#### 1. Data Processing & Simulation
We utilize the high-resolution ground truth from VisDrone to simulate realistic detection failures:
- **Random Drop**: 5-25% chance of missing a detection per frame.
- **Continuous Gaps**: Simulating long-term occlusions (2-10+ frames).

#### 2. Trajectory Reconstruction
Our system recovers lost detections through **Decision Fusion**:
- **Kalman Filter** maintains state across short gaps.
- **Spatio-Temporal Constraints** ensure physical plausibility (clamping velocity jumps and size deviations).

#### 3. ML-Based Association
Short-term tracklets are merged into global trajectories using an **Offline Tracklet Merging** strategy. We extract features like velocity consistency, centroid proximity, and bounding box overlap to train a classifier that validates potential merges.

---

### 🚀 Getting Started

#### Prerequisites
```bash
pip install -r extra/requirements-visdrone-pipeline.txt
```

#### Execution
1. Ensure the `VisDrone2019-MOT-train` dataset is in the root directory.
2. Open `offline_mot_tracking.ipynb` to run the baseline tracking and evaluation.

---

### 📈 Metrics & Results
Our system achieves competitive results on the VisDrone benchmark:
- **MOTA**: Multi-Object Tracking Accuracy
- **IDF1**: Identity F1 Score
- **Track Smoothness**: Reduced trajectory jitter through Kalman smoothing.

---

### 👥 Team - ML Code Crafters (Group 9)
*Dedicated to advancing precision in aerial surveillance and tracking.*
