---
title: AI-Powered Network Intrusion Detection
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: true
short_description: A real-time Random Forest NIDS dashboard for network traffic analysis.
---

# 🛡️ AI-Powered Network Intrusion Detection System (NIDS)

This dashboard uses a **Random Forest Algorithm** to monitor network traffic and classify it as safe (**Benign**) or a threat (**Malicious**). It provides an interactive way to train AI and test it against simulated network attacks.

## 🚀 Live Demo
For the best experience, use the direct link below to open the dashboard in **Full Screen** (no vibration):

👉 **[Launch AI NIDS Dashboard (Direct Link)](https://mohd-saad-ai-nids-dashboard.hf.space)**

---

## 🛠️ System Overview
* **AI Model:** Random Forest Classifier (Scikit-Learn).
* **Dataset Simulation:** Mimics the CIC-IDS2017 dataset patterns.
* **Infrastructure:** Deployed via **Docker** on Hugging Face Spaces.
* **UI Framework:** Streamlit for real-time visualization.

## 📁 Project Architecture

```text
.
├── .streamlit/
│   └── config.toml      # UI Stabilization Settings
├── app.py               # ML Logic & Dashboard Code
├── Dockerfile           # Container Environment
├── requirements.txt     # Python Dependencies
└── README.md            # Metadata & Documentation
