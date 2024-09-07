import streamlit as st
from pages_ import introduction, upload, cleaning, profiling, modeling, download, guide
from PIL import Image
import os


with st.sidebar:
    st.image(Image.open("images/AutoMl.png"))
    st.title("AutoML")
    choice = st.radio("Navigation",
                      ["Introduction", "Upload", "Preprocessing", "EDA Visuals", "Training",
                       "Saved Models","User Guide"])

if choice == "Introduction":
    introduction.show()
elif choice == "Upload":
    df, columns = upload.show()
elif choice == "Preprocessing":
    if not os.path.exists('./dataset.csv'):
        st.subheader("Please Upload your Dataset")
    else:
        cleaning.show() 
elif choice == "EDA Visuals":
    if not os.path.exists('./data.csv') and not os.path.exists('./dataset.csv') :
        st.subheader("Please upload your Dataset")
    else:
        profiling.show()
elif choice == "Training":
    if not os.path.exists('./data.csv'):
        st.subheader("Go to Preprocessing")
    else:
        modeling.show()
elif choice == "Saved Models":
    if not os.path.exists('./data.csv'):
        st.subheader("Go to Preprocessing")
    else:
        download.show()  
elif choice == "User Guide":
    guide.show()
