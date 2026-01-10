---
title: AI Powered Network Intrusion Detection System
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# AI-Powered Network Intrusion Detection System (NIDS)

This project uses a **Random Forest Classifier** to detect and classify network traffic as either "Benign" (Safe) or "Malicious" (Threat). It features an interactive dashboard for training the AI and simulating live traffic.

## 🚀 How to Run
Once deployed on Hugging Face, simply:
1. Go to the **App** tab.
2. Click **Train Model Now** in the sidebar.
3. View the **Performance Metrics** and use the **Traffic Simulator** to test custom inputs.

## 🛠️ Technology Stack
* **Language:** Python
* **ML Model:** Random Forest (Scikit-Learn)
* **Dashboard:** Streamlit
* **Deployment:** Docker / Hugging Face Spaces

## 📊 Dataset
The model is trained on synthetic data based on the **CIC-IDS2017** benchmark, including features like Flow Duration, Packet Length, and Active Time.