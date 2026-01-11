import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION & ANTI-VIBRATION LOCK ---
st.set_page_config(
    page_title="AI NIDS Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# This CSS block creates a "Fixed Container" for the app. 
# It stops the iframe from constantly trying to grow and shrink.
st.markdown(
    """
    <style>
    /* Stop the resizing loop between Streamlit and HF */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh;
        overflow: hidden;
    }
    .main {
        overflow-y: auto;
        height: 100vh;
    }
    /* Lock the sidebar to prevent width-shaking */
    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 300px;
    }
    /* Hide the top bar and footer which often trigger UI redraws */
    header, footer, #MainMenu {
        visibility: hidden;
    }
    /* Hide scrollbar for a cleaner look while keeping functionality */
    ::-webkit-scrollbar {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. DATA GENERATION ---
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
    # Patterning: Attacks have higher packet counts and lower durations
    df.loc[df['Label'] == 1, 'Total_Fwd_Packets'] += np.random.randint(50, 200, size=df[df['Label']==1].shape[0])
    df.loc[df['Label'] == 1, 'Flow_Duration'] = np.random.randint(1, 1000, size=df[df['Label']==1].shape[0])
    return df

df = load_data()

# --- 3. SIDEBAR & PREPROCESSING ---
st.sidebar.header("⚙️ System Settings")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("RF Tree Count", 10, 200, 100)

X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=42)

# --- 4. DASHBOARD UI ---
st.title("🛡️ AI-Powered Network Intrusion Detection")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. AI Training")
    if st.button("🚀 Train AI Model", use_container_width=True):
        with st.spinner("Analyzing patterns..."):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            st.session_state['nids_model'] = model
            st.success("Model Trained Successfully!")

with col2:
    st.subheader("2. Detection Metrics")
    if 'nids_model' in st.session_state:
        y_pred = st.session_state['nids_model'].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc*100:.2f}%")
        m2.metric("Threats Detected", int(np.sum(y_pred)))

        # Visual Chart
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Train the model to see performance data.")

# --- 5. LIVE TESTING ---
st.divider()
st.subheader("3. Live Packet Inspection")
c1, c2, c3, c4, c5 = st.columns(5)
p_port = c1.number_input("Port", 1, 65535, 80)
p_dur = c2.number_input("Duration", 0, 100000, 500)
p_pkts = c3.number_input("Packets", 0, 500, 10)
p_len = c4.number_input("Avg Len", 0, 1500, 100)
p_active = c5.number_input("Active", 0, 1000, 50)

if st.button("🔍 Analyze Traffic", use_container_width=True):
    if 'nids_model' in st.session_state:
        input_data = pd.DataFrame([[p_port, p_dur, p_pkts, p_len, p_active]], columns=X.columns)
        res = st.session_state['nids_model'].predict(input_data)
        if res[0] == 1:
            st.error("🚨 ALERT: Malicious Activity Detected!")
        else:
            st.success("✅ TRAFFIC STATUS: BENIGN")
    else:
        st.error("Train the model first!")
