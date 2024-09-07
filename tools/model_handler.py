import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

class ModelHandler:
    def __init__(self, df):
        self.df = df
        self.model_dict = {
            "LOR": "Logistic Regression", "DT": "Decision Trees", "RF": "Random Forest", "NB": "Naive Bayes",
            "SVM": "Support Vector Machines (SVM)", "GB": "Gradient Boosting", "NN": "Neural Networks",
            "QDA": "Quadratic Discriminant Analysis (QDA)", "AB": "Adaptive Boosting (AdaBoost)",
            "GP": "Gaussian Processes", "PT": "Perceptron", "KNC": "KNN Classifier", "RC": "Ridge Classifier",
            "PA": "Passive Aggressive Classifier", "EN": "Elastic Net", "LAR": "Lasso Regression",
            "LR": "Linear Regression", "PR": "Polynomial Regression", "SVR": "Support Vector Regression",
            "DTR": "Decision Tree Regression", "RFR": "Random Forest Regression", "RR": "Ridge Regression",
            "LASR": "Lasso Regression", "GR": "Gaussian Regression", "KNR": "KNN Regression", "ABR": "AdaBoost",
            "AP": "Affinity Propagation", "AC": "Agglomerative Clustering", "BC": "BIRCH", "DB": "DBSCAN",
            "KM": "K-Means", "MBK": "Mini-Batch K-Means", "MS": "Mean Shift", "OC": "OPTICS", "SC": "Spectral Clustering",
            "GMM": "Gaussian Mixture Model"
        }

    def download_model(self, model_name):
        if os.path.exists(f"./{model_name}.pkl"):
            with open(f"{model_name}.pkl", 'rb') as file:
                data = file.read()
            return data
        return None

    def make_prediction(self, model_name, input_data):
        with open(f"{model_name}.pkl", 'rb') as f:
            model = pickle.load(f)
        return model.predict(input_data)[0]

    def get_available_models(self, model_type):
        model_list = {
            "Regression": ['LR', 'PR', 'SVR', 'DTR', 'RFR', 'RR', 'LASR', 'GR', 'ABR', 'KNR'],
            "Classification": ['LOR', 'DT', 'RF', 'NB', 'SVM', 'GB', 'NN', 'QDA', 'AB', 'GP', 'PT', 'RC', 'PA', 'EN', 'LAR', 'KNC'],
            "Clustering": ['AP', 'AC', 'BC', 'DB', 'KM', 'MBK', 'MS', 'OC', 'SC', 'GMM']
        }
        return model_list.get(model_type, [])

    def get_model_label(self, model_code):
        return self.model_dict.get(model_code, "Unknown Model")