import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2e54ff;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-text {
        font-size: 24px;
        color: #0f5132;
        background-color: #d1e7dd;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .stSelectbox label, .stNumberInput label {
        color: #2c3e50;
        font-size: 16px;
        font-weight: 500;
    }
    .stButton button {
        background-color: #2e54ff;
        color: white;
        width: 100%;
        padding: 0.5rem 1rem;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and dataset
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model files not found. Please ensure 'pipe.pkl' and 'df.pkl' are in the correct directory.")
    st.stop()

# Title
st.title('💻 Laptop Price Predictor')

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏢 Basic Information")
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])
    weight = st.number_input('Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, step=0.1)

with col2:
    st.markdown("### 🖥️ Display Details")
    touchScreen = st.selectbox('Touchscreen', ['No','Yes'])
    ips = st.selectbox('IPS Display', ['No','Yes'])
    screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1)
    resolution = st.selectbox('Screen Resolution',
        ['1920x1080', '1366x768', '1600x900', '3840x2160',
         '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

with col3:
    st.markdown("### 💽 Hardware Specifications")
    cpu = st.selectbox('CPU Brand', df['CPU_Brand'].unique())
    hdd = st.selectbox('HDD Storage (GB)', [0,128,256,512,1024,2048])
    ssd = st.selectbox('SSD Storage (GB)', [0,8,128,256,512,1024])
    gpu = st.selectbox('GPU Brand', df['GPU_Brand'].unique())
    os = st.selectbox('Operating System', df['os'].unique())

# Center the predict button
col1, col2, col3 = st.columns([1,1,1])
with col2:
    predict_button = st.button("Predict Price 🔍")

if predict_button:
    try:
        # Convert touchscreen and IPS to binary
        touchScreen = 1 if touchScreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Calculate PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

        # Create query array
        query = np.array([company,type,ram,weight,touchScreen,ips,ppi,cpu,hdd,ssd,gpu,os])
        query = query.reshape(1,12)

        # Make prediction
        prediction = np.exp(pipe.predict(query))[0]

        # Display prediction with formatting
        st.markdown(f"""
            <div class="prediction-text">
                Predicted Laptop Price: ₹{prediction:,.2f}
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check all input fields and try again.")

# Add helpful information at the bottom
st.markdown("""
---
### 📌 How to Use
1. Fill in all the specifications of the laptop
2. Click on 'Predict Price' to get the estimated price
3. The prediction is based on historical laptop data and prices are in Indian Rupees (₹)

### 💡 Tips
- Higher RAM, better CPU/GPU, and storage typically result in higher prices
- Screen resolution and size significantly impact the price
- Gaming laptops generally cost more than regular laptops
""")
