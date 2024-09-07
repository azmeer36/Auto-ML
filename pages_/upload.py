import streamlit as st
import pandas as pd
import io

def show():
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        describe_table = df.describe()
        if len(describe_table)>0:
            minmax = {}
            for i in describe_table:
                minmax[i] = [describe_table[i]['min'], describe_table[i]['max']]
        st.session_state['minmaxtable'] = minmax
        columns = df.columns
        
        st.subheader("Shape and size of the data")
        st.write(df.shape)
        st.write(df.size)
        
        st.subheader("Data Description ")
        st.dataframe(describe_table)
        
        st.subheader("Do you want information about data ")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("Duplicates")
        st.write("Number of duplicate rows: ", df.duplicated().sum())
            
        st.subheader("Missing values")
        
        st.write("Number of missing rows")
        st.dataframe(df.isnull().sum())
            
        return df, columns
    return None, None
            
    
            
        
        