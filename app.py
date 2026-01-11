import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION & ANTI-VIBRATION CSS ---
st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

# This CSS block kills the "auto-resize loop" between Streamlit and the HF Iframe
st.markdown(
    """
    <style>
    /* Force the app to be a fixed height to stop the shaking */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh;
        overflow: hidden;
    }
    .main {
        overflow-y: auto;
    }
    /* Hide the flickering elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Optimize the block container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. TITLE AND DESCRIPTION ---
st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.markdown("Analyze network traffic and classify threats using Random Forest.")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 5000
    data = {
        'Destination_Port': np.random.randint(1, 65535, n_samples),
        'Flow_Duration': np.random.randint(100, 100000, n_samples),
        'Total_Fwd_Packets': np.random.randint(1, 100, n_samples),
        'Packet_Length_Mean': np.random.uniform(10, 1500, n_samples),
        'Active_Mean': np.random.uniform(0, 1000, n_samples),
        'Label': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    # Simulate attack patterns
    df.loc[df['Label'] == 1, 'Total_Fwd_Packets'] += np.random.randint(50, 200, size=df[df['Label']==1].shape[0])
    df.loc[df['Label'] == 1, 'Flow_Duration'] = np.random.randint(1, 1000, size=df[df['Label']==1].shape[0])
    return df

df = load_data()

# Sidebar
st.sidebar.header("🛠️ Settings")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Forest Trees", 10, 200, 100)

# --- 4. PREPROCESSING ---
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=42)

# --- 5. TRAINING & EVALUATION ---
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Training")
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("Learning patterns..."):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['model'] = model
            st.success("Model Ready!")

with col2:
    st.subheader("2. Performance")
    if 'model' in st.session_state:
        y_pred = st.session_state['model'].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        m1, m2 = st.columns(2)
        m1.metric("Accuracy Score", f"{acc*100:.2f}%")
        m2.metric("Threats Detected", int(np.sum(y_pred)))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

# --- 6. LIVE SIMULATOR ---
st.divider()
st.subheader("3. Live Network Traffic Simulator")
c1, c2, c3, c4, c5 = st.columns(5)
p_port = c1.number_input("Dest Port", 1, 65535, 80)
p_dur = c2.number_input("Duration", 0, 100000, 500)
p_pkts = c3.number_input("Packets", 0, 500, 10)
p_len = c4.number_input("Avg Len", 0, 1500, 100)
p_active = c5.number_input("Active Time", 0, 1000, 50)

if st.button("🔍 Run Security Scan", use_container_width=True):
    if 'model' in st.session_state:
        input_data = pd.DataFrame([[p_port, p_dur, p_pkts, p_len, p_active]], columns=X.columns)
        res = st.session_state['model'].predict(input_data)
        if res[0] == 1:
            st.error("🚨 ALERT: Malicious Activity Detected!")
        else:
            st.success("✅ Traffic is Clean.")
    else:
        st.error("Please train the model first.")
