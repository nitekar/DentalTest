"""
Streamlit UI for Dental X-Ray Classification System.
Provides interactive interface for predictions, visualizations, and retraining.
"""

import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import numpy as np


# Page configuration
st.set_page_config(
    page_title="Dental X-Ray Classifier",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "https://dentaltest-1.onrender.com"


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_image(image_file):
    """Send image to API for prediction."""
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


def upload_bulk_data(zip_file):
    """Upload bulk training data."""
    try:
        files = {"file": zip_file}
        response = requests.post(f"{API_URL}/upload_bulk", files=files, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def trigger_retraining():
    """Trigger model retraining."""
    try:
        response = requests.post(f"{API_URL}/retrain", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Retraining error: {str(e)}")
        return None


def get_retrain_status():
    """Get retraining status."""
    try:
        response = requests.get(f"{API_URL}/retrain_status", timeout=5)
        return response.json()
    except:
        return None


def plot_class_distribution():
    """Plot class distribution from training data."""
    train_dir = Path("data/train")
    
    if not train_dir.exists():
        st.warning("Training data not found")
        return
    
    class_counts = {}
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
            class_counts[class_dir.name] = count
    
    if not class_counts:
        st.warning("No training data found")
        return
    
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    
    fig = px.bar(
        df,
        x='Class',
        y='Count',
        title='Training Data Distribution',
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_title="Dental Condition",
        yaxis_title="Number of Images",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_confidence_gauge(confidence):
    """Plot confidence as gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def plot_probability_distribution(probabilities):
    """Plot all class probabilities."""
    df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(
        df,
        x='Probability',
        y='Class',
        orientation='h',
        title='Class Probability Distribution',
        color='Probability',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Probability",
        yaxis_title="Dental Condition",
        showlegend=False
    )
    
    return fig


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header"> Dental X-Ray Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f9b7.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Go to",
            [" Home", " Prediction", " Data Insights", " Retraining"]
        )
        
        st.markdown("---")
    
    # Home Page
    if page == " Home":
        st.header("Welcome to the Dental X-Ray Classification System")
        
        st.write("""
        This application uses advanced machine learning to analyze dental X-ray images 
        and classify various dental conditions. Upload your X-ray images to get instant 
        predictions and insights.
        
        **Key Features:**
        - Instant X-ray image classification
        - Multiple dental condition detection
        - Model retraining capabilities
        - Data visualization and insights
        
        Navigate through the sidebar to explore different features of the application.
        """)
    
    # Prediction Page
    elif page == " Prediction":
        st.header("Upload X-Ray for Classification")
        
        uploaded_file = st.file_uploader(
            "Choose a dental X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a dental X-ray image for classification"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            with col2:
                # Predict button
                if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing X-ray..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        
                        # Get prediction
                        result = predict_image(uploaded_file)
                        
                        if result:
                            # Display prediction
                            prediction = result['prediction']
                            
                            st.markdown(f"### Prediction: **{prediction}**")
                            
                            # Probability distribution
                            st.subheader("All Class Probabilities")
                            fig = plot_probability_distribution(result['all_probabilities'])
                            st.plotly_chart(fig, use_container_width=True)
    
    # Data Insights Page
    elif page == " Data Insights":
        st.header("Training Data Insights")
        
        tab1, tab2 = st.tabs([" Class Distribution", " Statistics"])
        
        with tab1:
            st.subheader("Training Data Distribution")
            plot_class_distribution()
        
        with tab2:
            st.subheader("Dataset Statistics")
            
            train_dir = Path("data/train")
            test_dir = Path("data/test")
            
            if train_dir.exists():
                train_count = sum(len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png'))) 
                                for d in train_dir.iterdir() if d.is_dir())
                
                test_count = 0
                if test_dir.exists():
                    test_count = sum(len(list(d.glob('*.jpg'))) + len(list(d.glob('*.png'))) 
                                   for d in test_dir.iterdir() if d.is_dir())
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training Images", train_count)
                
                with col2:
                    st.metric("Test Images", test_count)
                
                with col3:
                    st.metric("Total Images", train_count + test_count)
                
                # Class-wise breakdown
                st.markdown("---")
                st.subheader("Class-wise Breakdown")
                
                class_data = []
                for class_dir in train_dir.iterdir():
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                        class_data.append({
                            'Class': class_dir.name,
                            'Images': count,
                            'Percentage': f"{(count/train_count*100):.1f}%"
                        })
                
                df = pd.DataFrame(class_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No training data found")
    
    # Retraining Page
    elif page == " Retraining":
        st.header("Model Retraining")
        

        
        # Upload section
        st.subheader(" Upload Training Data")
        
        uploaded_file = st.file_uploader(
            "Upload training image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a dental X-ray image for training"
        )
        
        class_name = st.selectbox(
            "Select class for this image",
            ["BDC_BDR", "Caries", "Fractured", "Healthy", "Impacted", "Infection"]
        )
        
        if uploaded_file is not None:
            if st.button(" Upload Data", type="primary"):
                with st.spinner("Uploading image..."):
                    uploaded_file.seek(0)
                    
                    files = {"file": uploaded_file}
                    data = {"class_name": class_name}
                    
                    try:
                        response = requests.post(f"{API_URL}/upload_bulk", files=files, data=data, timeout=120)
                        response.raise_for_status()
                        result = response.json()
                        
                        if result:
                            st.success(f" {result['message']}")
                            
                            # Show prediction result if available
                            if result.get('prediction'):
                                pred = result['prediction']['prediction']
                                st.info(f" Model predicts: **{pred}**")
                            else:
                                st.info(f" Added to class: {class_name}")
                            
                            # Automatically trigger retraining
                            with st.spinner("Starting retraining..."):
                                try:
                                    retrain_response = requests.post(f"{API_URL}/retrain", json={"force": True}, timeout=60)
                                    retrain_response.raise_for_status()
                                    retrain_result = retrain_response.json()
                                    st.success("Retraining started automatically!")
                                except Exception as retrain_error:
                                    st.warning(f"Upload successful, but retraining failed: {str(retrain_error)}")
                    except Exception as e:
                        st.error(f"Upload error: {str(e)}")
        

        



if __name__ == "__main__":
    main()