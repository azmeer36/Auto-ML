import streamlit as st
from PIL import Image

def show():
    
    st.markdown("""
    AutoMl offers a full range of tools and features designed to support your data-driven projects. Whether you're a data hobbyist, business analyst, or machine learning expert, our platform simplifies your workflow, enabling you to derive valuable insights from your data.
    
    ### *Upload:*
    Easily upload your dataset and begin exploring it in detail. Once uploaded, our platform generates a comprehensive description of the data, including key statistics like the number of rows, columns, and unique values. You can quickly assess data quality by identifying any duplicate entries or missing values across columns.
    """)
    
    st.image(Image.open("images/upload.png"))
    st.markdown("""
    ### *Preprocessing:*
    This section empowers you to enhance the quality of your data effortlessly. If there are any unnecessary columns in your dataset, you can easily remove them to focus on the relevant information. Additionally, you have the option to remove duplicates and handle missing values intelligently. Our platform will assist you in maintaining a clean and accurate dataset for further analysis.""")
    st.image(Image.open("images/clean1.png"))
    
    st.markdown("""
    ### *EDA Visuals:*
    The EDA (Exploratory Data Analysis) feature, powered by pandas profiling, provides an in-depth look at your dataset. You’ll gain a thorough understanding of data types, summary statistics, and variable correlations, allowing you to uncover patterns and relationships within your data.""")
    st.image(Image.open("images/eda1.png"))
    
    st.markdown("""
    ### *Training:*
    For the data scientists and machine learning enthusiasts, the ML model training section is a treasure trove of possibilities. Begin by selecting the target variable and specifying the problem type - be it regression, classification, clustering, or dimensionality reduction. Then, fine-tune the hyperparameters of various algorithms to obtain the best results. Once the models are trained, our platform will present a detailed performance summary, including essential metrics such as MAE, RMSE, R2 score, precision, recall, and F1 score. You'll be empowered to make data-driven decisions with confidence.""")
    col4, col5 = st.columns(2)
    with col4:
        st.image(Image.open("images/modelling1.png"))

    with col5:
        st.image(Image.open("images/modelling2.png"))


    st.markdown("""
    ### *Download:*
    At the end of your journey, you can seamlessly download your model in the form of convenient pickle files, enabling you to deploy your trained model easily. Additionally, if you need a copy of your processed dataset, you can download it in the universally-accepted CSV format, ensuring your data remains accessible for further analysis or sharing.""")
    st.image(Image.open("images/download.png"))
    st.markdown("""
    Join us in exploring the power of data analysis and machine learning! Our platform is your gateway to uncovering hidden insights, making data-driven decisions, and bringing your projects to new heights. Let's embark on this exciting journey together!""")
