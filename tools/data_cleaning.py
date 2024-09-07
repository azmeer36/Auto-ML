import os
import streamlit as st
import numpy as np
from scipy.stats import zscore
import pandas as pd

numerical = ['int64', 'float64','float','int']
categorical = ['object','string']

class DataCleaning:
    def __init__(self, df):
        self.df = df

    def remove_columns(self, remove_cols):
        if remove_cols:
            self.df.drop(remove_cols, axis=1, inplace=True)
            
    def missing_cols(self):
        missing_columns = self.df.columns[self.df.isnull().any()].tolist()
        return missing_columns

    def fill_missing_values(self, filling_col):
        for column in filling_col:
            dt = str(self.df[column].dtype)

            if dt in numerical:
                option = st.radio(
                    f"Fill missing values in **{column}** with",
                    ("Constant", "Mean", "Median"),
                    key=f"fill_option_{column}",
                    help="Choose the method to fill missing numeric values"
                )
            elif dt in categorical:
                option = st.radio(
                    f"Fill missing values in **{column}** with",
                    ("Constant", "Mode"),
                    key=f"fill_option_{column}",
                    help="Choose the method to fill missing categorical values"
                )

            if option == "Constant":
                con = st.text_input(f"Enter the filling constant for {column}")
                self.df[column].fillna(con, inplace=True)
            elif option == 'Mean':
                self.df[column].fillna(self.df[column].mean(), inplace=True)
            elif option == 'Median':
                self.df[column].fillna(self.df[column].median(), inplace=True)
            elif option == 'Mode':
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)

        st.write(self.df)

    def detect_outliers(self, outlier_columns):
        for column in outlier_columns:
            option = st.selectbox(
                f"Outliers detection method for **{column}**",
                ('Z-Score', 'IQR', 'Percentile'),
                help="Choose the method for outlier detection"
            )

            if option == "Z-Score":
                trimming_option = st.radio(
                    f"Z-Score outlier detection method for **{column}**",
                    ("Trimming", "Capping"),
                    key=f"zscore_option_{column}",
                    help="Choose the method for outlier detection"
                )

                z_scores = zscore(self.df[column])
                if trimming_option == "Trimming":
                    self.df = self.df[(np.abs(z_scores) < 3)]
                elif trimming_option == "Capping":
                    lower_threshold = -3
                    upper_threshold = 3
                    self.df[column] = np.where(self.df[column] < lower_threshold, lower_threshold, self.df[column])
                    self.df[column] = np.where(self.df[column] > upper_threshold, upper_threshold, self.df[column])

            elif option == "IQR":
                q1 = self.df[column].quantile(0.25)
                q3 = self.df[column].quantile(0.75)
                iqr = q3 - q1
                lower_threshold = q1 - 1.5 * iqr
                upper_threshold = q3 + 1.5 * iqr

                trimming_option = st.radio(
                    f"IQR outlier detection method for {column}",
                    ("Trimming", "Capping"),
                    key=f"iqr_option_{column}",
                    help="Choose the method for outlier detection"
                )
                if trimming_option == "Trimming":
                    self.df = self.df[self.df[column].between(lower_threshold, upper_threshold)]
                elif trimming_option == "Capping":
                    self.df[column] = np.where(self.df[column] < lower_threshold, lower_threshold, self.df[column])
                    self.df[column] = np.where(self.df[column] > upper_threshold, upper_threshold, self.df[column])

            elif option == "Percentile":
                lower_percentile = st.number_input(f"Enter the lower percentile for {column} (0-50)", value=5, min_value=0, max_value=50)
                upper_percentile = st.number_input(f"Enter the upper percentile for {column} (50-100)", value=95, min_value=50, max_value=100)
                lower_limit = self.df[column].quantile(lower_percentile / 100)
                upper_limit = self.df[column].quantile(upper_percentile / 100)

                trimming_option = st.radio(
                    f"Percentile outlier detection method for {column}",
                    ("Trimming", "Capping"),
                    key=f"percentile_option_{column}",
                    help="Choose the method for outlier detection"
                )
                if trimming_option == "Trimming":
                    self.df = self.df[self.df[column].between(lower_limit, upper_limit)]
                elif trimming_option == "Capping":
                    self.df[column] = np.where(self.df[column] < lower_limit, lower_limit, self.df[column])
                    self.df[column] = np.where(self.df[column] > upper_limit, upper_limit, self.df[column])

        st.write(self.df)


    def save_to_csv(self, filename="data.csv"):
        self.df.to_csv(filename, index=False)