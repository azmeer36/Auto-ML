import streamlit as st
import os
import pandas as pd
import numpy as np
from tools.model_handler import ModelHandler 

def show():
    df = pd.read_csv("./data.csv")
    df = df.iloc[:, :]

    # Save the dataset for download
    df.to_csv("DATA.csv", index=False)
    with open('DATA.csv', 'rb') as file:
        data = file.read()

    st.download_button(
        label="Download Final Dataset",
        data=data,
        file_name='DATA.csv',
        mime='text/csv',
    )
    st.markdown("---")

    model_handler = ModelHandler(df)
    st.write("#### Select Model Category: ")
    choicel = st.selectbox("", ["Regression", "Classification", "Clustering"])
    st.write("##### Avaialable Models (Click to download):")


    model_codes = model_handler.get_available_models(choicel)
    
    available_models = []

    if choicel in ["Regression", "Classification"]:
        for model_code in model_codes:
            model_data = model_handler.download_model(model_code)
            if model_data:
                available_models.append(model_code)
                st.download_button(
                    label=model_handler.get_model_label(model_code),
                    data=model_data,
                    file_name=f"{model_code}.pkl"
                )

        st.write("##### Prediction: ")
        if "chosen_target" in st.session_state:
            chosen_target = st.session_state['chosen_target']
            cols = df.columns
            predict = []
            for col in cols:
                if col == chosen_target:
                    continue
                if col in st.session_state['numerical_col_set']:
                    mv = st.session_state['minmaxtable'][col]
                    x = st.number_input(f"**{col}**")
                    v = (x - mv[0]) / (mv[1] - mv[0])
                    predict.append(v)
                elif col in st.session_state['categorical_col_set']:
                    uniquevals = df[col].unique()
                    x = st.selectbox(f"**{col}**", uniquevals)
                    v = [1 if i == x else 0 for i in uniquevals]
                    predict.extend(v)

            npredict = np.array(predict).reshape(1, -1)
            model_name = st.selectbox("Select the model to predict", available_models)
            st.write("Selected Model is ", model_handler.get_model_label(model_name))
            prediction = model_handler.make_prediction(model_name, npredict)
            st.subheader("Predicted value: " + str(prediction))
        else:
            st.error("Select the chosen target in Training page", icon="🚨")

    elif choicel == "Clustering":
        for model_code in available_models:
            model_data = model_handler.download_model(model_code)
            if model_data:
                st.download_button(
                    label=model_handler.get_model_label(model_code),
                    data=model_data,
                    file_name=f"{model_code}.pkl"
                )