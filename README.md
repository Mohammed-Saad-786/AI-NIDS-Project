---
title: AI-Powered Network Intrusion Detection System
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8501
pinned: true
short_description: A machine learning dashboard to detect network attacks in real-time.
---

# 🛡️ AI-Powered Network Intrusion Detection System (NIDS)

This project is an AI-driven security dashboard designed to monitor and classify network traffic. Using a **Random Forest Classifier**, the system distinguishes between safe (**Benign**) and malicious (**Attack**) traffic patterns.

## 🚀 Live Demo
You can access the stable version of the dashboard here:
[Direct Link (Zero-Vibration)](https://huggingface.co/spaces/mohd-saad/ai-nids-dashboard?embed=true)

---

## 🛠️ Key Features
* **Real-Time Classification:** Instantly classifies network packets based on traffic features.
* **Interactive Training:** Users can adjust training data size and model complexity via the sidebar.
* **Performance Metrics:** Displays Accuracy, Confusion Matrix, and Detection counts.
* **Attack Simulator:** Manual input section to test specific "what-if" network scenarios.

## 🧪 Technical Stack
* **Language:** Python 3.9
* **UI Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Handling:** Pandas & NumPy
* **Visualization:** Seaborn & Matplotlib
* **Deployment:** Docker & Hugging Face Spaces

## 📂 Project Structure
```text
AI_NIDS_Project/
├── .streamlit/          
│   └── config.toml      # Server and UI stabilization settings
├── app.py               # Main Streamlit application and ML logic
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md            # Metadata and documentation.

