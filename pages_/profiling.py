import os
import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

def show():
    # Determine which file to use
    if os.path.exists('./data.csv'):
        file_path = './data.csv'
    elif os.path.exists('./dataset.csv'):
        file_path = './dataset.csv'
        
    # Load the DataFrame
    df = pd.read_csv(file_path)
    df = df.iloc[:, 1:]  # Remove the first column if necessary
    st.title("Exploratory Data Analysis")

    # Button to generate and download the report
    if st.button("Generate Report"):
        # Generate the profile report
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
        report_file = "profiling_report.html"
        profile.to_file(report_file)  # Save the report as an HTML file
        with open(report_file, "rb") as f:
            st.download_button("Download Report", f, file_name=report_file, mime="text/html")