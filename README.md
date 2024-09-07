# **Auto ML** 🎯

Welcome to the **Auto ML** project, an all-in-one solution for automating the entire Machine Learning workflow, from data preprocessing to model deployment. This tool is designed to empower users with little or no coding experience to efficiently create machine learning models, generate insights, and make predictions. Auto ML simplifies the complexity of Machine Learning into a user-friendly, step-by-step process.
![modelling1](https://github.com/user-attachments/assets/5ab3d3ce-efb1-484f-b652-1764c151c0db)

## **Table of Contents** 📚
- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Key Steps](#key-steps)
  - [1. Data Uploading](#1-data-uploading)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Data Profiling](#3-data-profiling)
  - [4. Data Visualization](#4-data-visualization)
  - [5. Data Modeling](#5-data-modeling)
  - [6. Model Download](#6-model-download)
- [How It Works](#how-it-works)
- [Contribution](#contribution)
- [License](#license)

## **Introduction** 📊

The **Auto ML** project is designed to democratize the power of Machine Learning by automating the most complex aspects of the process. With Auto ML, users can upload data, clean it, visualize it, and train models without writing a single line of code. The platform provides a seamless interface to build machine learning models for regression and classification tasks, complete with the ability to download and deploy trained models for prediction.

## **Features** ✨

- **Interactive Data Upload**: Upload datasets in CSV or XLSX formats and get instant feedback about the dataset's structure and content.
- **Automated Data Cleaning**: Effortlessly clean your data by removing irrelevant columns, handling missing values, and detecting/removing outliers.
- **Comprehensive Data Profiling**: Generate detailed reports with key insights into data distributions, variable types, and statistical summaries.
- **Interactive Visualizations**: Explore your data visually using a Tableau-like interface to uncover patterns and relationships.
- **Machine Learning Model Building**: Train models for both classification and regression tasks with a wide selection of algorithms, including Decision Trees, Random Forest, SVM, and more.
- **Model Evaluation**: Get detailed evaluation metrics like R2, RMSE, precision, recall, and F1 scores to assess model performance.
- **Model Export**: Save your trained models in Pickle format for easy deployment, and download processed data in CSV format.

## **Tech Stack** 🛠️

- **Backend**: Python
- **Frontend**: Streamlit (for interactive UI)
- **Machine Learning**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Pandas Profiling
- **Data Visualization**: Plotly, Altair, Seaborn
- **Deployment**: Streamlit Cloud, GitHub

## **Installation** 🚀

To get the Auto ML project up and running locally, follow these steps:

1. **Clone the repository** from GitHub:
    ```bash
    git clone https://github.com/azmeer36/Auto_Ml.git
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run main.py
    ```

4. Open your browser and navigate to the local URL provided by Streamlit (typically `http://localhost:8501`).

## **Usage** 🧑‍💻

The application is accessible through a web interface, making it easy for users of all technical backgrounds to interact with the system. Follow these simple steps:

1. Upload your dataset in either `.csv` or `.xlsx` format.
2. Configure various preprocessing options, such as cleaning the data, handling missing values, and more.
3. Train machine learning models on your dataset by choosing the appropriate algorithm and hyperparameters.
4. Download your trained model and processed data for deployment or further analysis.

## **Key Steps** 🔍

### 1. **Data Uploading** 🚀

- **Format Supported**: CSV and XLSX
- **Overview**: Users are presented with an interactive UI to upload their datasets. Once uploaded, the platform displays detailed information about the dataset, including:
  - Dataset shape and size
  - Data types of each column
  - Descriptive statistics: Mean, Median, Min, Max, etc.
  - Presence of missing values and duplicate rows

### 2. **Data Cleaning** 🧼

- **Features**:
  - Remove unwanted columns
  - Handle missing values by filling them with mean, median, or constant values
  - Detect and remove duplicate rows
  - Outlier detection and handling using Z-score, IQR, and percentile-based methods
- **Why It Matters**: Cleaning data is crucial for ensuring the quality and accuracy of machine learning models.

### 3. **Data Profiling** 📋

- **Automated Report**: Auto ML generates a detailed profiling report for your dataset, which includes:
  - Data distributions
  - Descriptive statistics for numerical columns
  - Distribution of categorical variables
  - Correlations between variables
  - Missing value heatmaps
- **Purpose**: Profiling helps users understand the structure and health of their data before proceeding with modeling.

### 4. **Data Visualization** 📊

- **Interactive Plots**: Auto ML offers interactive data visualization capabilities, similar to Tableau, allowing users to:
  - Generate bar charts, histograms, scatter plots, and heatmaps
  - Explore relationships between variables
  - Identify patterns and trends in the data

### 5. **Data Modeling** 🤖

- **Supervised Machine Learning**: Users can select their target variable, choose the problem type (regression or classification), and pick algorithms to train on the dataset.
  - **Regression Algorithms**: Linear Regression, Polynomial Regression, KNN Regression, Decision Tree, Random Forest, Naive Bayes, etc.
  - **Classification Algorithms**: Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, Support Vector Machine, AdaBoost, etc.
- **Model Evaluation Metrics**:
  - **Regression**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R2 Score
  - **Classification**: Precision, Recall, F1 Score, Accuracy
- **Hyperparameter Tuning**: Fine-tune the model's hyperparameters for optimal performance.

### 6. **Model Download** 💾

- **Exporting Models**: Once your model has been trained and evaluated, you can download it as a Pickle file for later use.
- **Processed Data**: You can also download the cleaned and preprocessed data in CSV format for further analysis or sharing with collaborators.

## **How It Works** ⚙️

Auto ML automates the end-to-end machine learning process:
1. **Data Upload and Validation**: The system validates the uploaded dataset, ensuring it’s ready for analysis.
2. **Automated Data Cleaning**: Outliers, duplicates, and missing values are handled intelligently.
3. **Profiling and Visualization**: Provides insights into the data structure and relationships through comprehensive reports and interactive visualizations.
4. **Model Building**: Automatically trains models using selected algorithms and displays evaluation metrics for each model.
5. **Deployment Ready**: Allows you to download your models and deploy them for predictions on new data.

## **Contribution** 🤝

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request. You can also open issues if you find bugs or want to request new features.

To contribute:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## **License** 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy coding! 😊
