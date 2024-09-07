import os
import streamlit as st
import numpy as np
from scipy.stats import zscore
import pandas as pd
from tools.data_cleaning import DataCleaning


def show(): 
    df = pd.read_csv("./dataset.csv")
    data_cleaner = DataCleaning(df)

    st.subheader("Removing Columns:")
    remove_cols = st.multiselect("Select Columns to remove:", df.columns)
    if st.button("Process", key="remove col"):
        if remove_cols:
            data_cleaner.remove_columns(remove_cols)
            st.dataframe(data_cleaner.df.head(5))
        else:
            st.write("Please select columns to remove")          
    st.markdown("---")
        
        
    st.subheader("Removing Duplicate Rows:")
    if st.button("Process", key='remove duplicate'):
        data_cleaner.df.drop_duplicates(inplace=True)
        st.write("Number of duplicate rows: ", data_cleaner.df.duplicated().sum()) 
    st.markdown("---")


    st.subheader("Filling Missing Values:")
    missing_cols = data_cleaner.missing_cols()
    if missing_cols:
        st.write("##### Columns with Missing Values:")
        st.markdown(", ".join(missing_cols))  # Display the column names as a comma-separated list
    else:
        st.write("##### No missing values in the dataset.")
    filling_col = st.multiselect("Select columns to fill missing values", missing_cols)
    # if st.button("Process", key="fill col"):
    if filling_col:
        data_cleaner.fill_missing_values(filling_col)
    st.markdown("---")
    
            
    st.subheader("Outlier Detection:")
    outlier_columns = st.multiselect("Select columns for outlier detection", data_cleaner.df.columns)
    # if st.button("Process", key="remove outlier"):
    if outlier_columns:
        data_cleaner.detect_outliers(outlier_columns)
    st.markdown("---")


    if st.button("Save Cleaned Data"):
        data_cleaner.save_to_csv()
        st.success("Data saved to data.csv")
