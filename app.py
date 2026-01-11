import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

# Stability Fix: Prevent the "vibration" loop on Hugging Face
st.markdown(
    """
    <style>
    .main { overflow-x: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    /* Fix for small screens/iframes */
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom Title and Description
st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.markdown("""
This system uses a **Random Forest Algorithm** to analyze network traffic patterns and classify them as safe (**Benign**) or a threat (**Malicious**).
""")

# --- 2. DATA LOADING (Simulation Mode) ---
@st.cache_data
def load_data():
    """
    Generates a synthetic dataset mimicking network traffic (CIC-IDS2017 style).
    """
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'Destination_Port': np.random.randint(1, 65535, n_samples),
        'Flow_Duration': np.random.randint(100, 100000, n_samples),
        'Total_Fwd_Packets': np.random.randint(1, 100, n_samples),
        'Packet_Length_Mean': np.random.uniform(10, 1500, n_samples),
        'Active_Mean': np.random.uniform(0, 1000, n_samples),
        'Label': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) # 0=Safe, 1=Attack
    }
    df = pd.DataFrame(data)
    
    # Logic for AI to learn: Malicious traffic usually has high packets & short duration
    df.loc[df['Label'] == 1, 'Total_Fwd_Packets'] += np.random.randint(50, 200, size=df[df['Label']==1].shape[0])
    df.loc[df['Label'] == 1, 'Flow_Duration'] = np.random.randint(1, 1000, size=df[df['Label']==1].shape[0])
    return df

df = load_data()

# Sidebar Controls
st.sidebar.header("🛠️ Model Configuration")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
st.sidebar.markdown("---")
st.sidebar.write("Dataset Size: 5,000 Samples")

# --- 3. PREPROCESSING & SPLIT ---
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=42)

# --- 4. MODEL TRAINING ---
st.divider()
col_train, col_metrics = st.columns([1, 2])

with col_train:
    st.subheader("1. Train the Brain")
    if st.button("🚀 Start Training", use_container_width=True):
        with st.spinner("Training Random Forest..."):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.success("Training Complete!")

    if 'model' in st.session_state:
        st.info("✅ Model is active and ready.")

# --- 5. EVALUATION METRICS ---
with col_metrics:
    st.subheader("2. Performance Analytics")
    if 'model' in st.session_state:
        model = st.session_state['model']
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy Score", f"{acc*100:.2f}%")
        m2.metric("Samples Tested", len(y_test))
        m3.metric("Attacks Caught", int(np.sum(y_pred)))

        # Visualization
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title("Detection Accuracy (Confusion Matrix)")
        st.pyplot(fig)
    else:
        st.warning("Waiting for model training...")

# --- 6. LIVE TRAFFIC SIMULATOR ---
st.divider()
st.subheader("3. Live Traffic Simulator")
st.write("Input packet details to test the AI's detection capability.")

c1, c2, c3, c4, c5 = st.columns(5)
p_port = c1.number_input("Dest Port", 1, 65535, 80)
p_dur = c2.number_input("Duration (ms)", 0, 100000, 500)
p_pkts = c3.number_input("Packets Count", 0, 500, 10)
p_len = c4.number_input("Avg Packet Len", 0, 1500, 100)
p_active = c5.number_input("Active Time", 0, 1000, 50)

if st.button("🔍 Analyze Network Traffic", use_container_width=True):
    if 'model' in st.session_state:
        # Align features with training data columns
        input_data = pd.DataFrame([[p_port, p_dur, p_pkts, p_len, p_active]], columns=X.columns)
        prediction = st.session_state['model'].predict(input_data)
        
        if prediction[0] == 1:
            st.error("🚨 ALERT: MALICIOUS TRAFFIC DETECTED!")
            st.toast("Security Alert Generated!")
        else:
            st.success("✅ TRAFFIC STATUS: BENIGN (SAFE)")
    else:
        st.error("Please train the model first using the button above!")
