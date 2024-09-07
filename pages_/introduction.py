import streamlit as st
from PIL import Image
from pages_ import guide

def show():
    st.write("# Welcome to AutoML! 👋")
    
    # Introduction text
    st.markdown("""
        This platform is designed to help you explore various machine learning techniques and models. 
        Whether you are a beginner or an experienced data scientist, you will find tools and resources 
        to assist you in your machine learning journey.

        ### Features:
        - **Supervised Learning**: Train and evaluate models for classification and regression tasks.
        - **Unsupervised Learning**: Explore clustering and dimensionality reduction techniques.
        - **Interactive Visualizations**: Visualize your data and model performance in real-time.
        - **User-Friendly Interface**: Easy navigation and intuitive controls for a seamless experience.

        For a comprehensive guide on how to use this platform, please refer to the User Guide:
        """)
      
    

    