# _Auto ML_

Hi, Thanks for your interest.

# Introduction:
 This project aims to automate the entire process of Machine Learning, making it accessible to users who may find coding challenging. Auto ML streamlines the process into 6 key steps, from data uploading to model deployment and prediction.

# Steps:

### Data Uploading:

Users are presented with an interactive interface to upload their dataset in CSV format.
The interface displays comprehensive details about the uploaded data, including dataset overview, shape, size, data types, descriptive statistics (mean, median, min, max), missing values, and duplicate rows.
### Data Cleaning:

Users can perform data cleaning tasks such as removing unwanted columns, filling missing values (with mean, median, or constant), removing duplicates, and outlier detection and treatment using methods like Z-score, IQR, and Percentile.
### Profiling:

Generates a report providing insights into the data, including basic visualizations, and distributions of columns.
### Data Visualization:

Offers a Tableau-like interface for exploring data through visualizations, enabling users to derive meaningful insights.
### Data Modeling:

Supervised Machine Learning: Users select problem type (regression or classification), target column, and algorithms to train on data.
Algorithms available for regression include linear regression, polynomial regression, KNN Regression, Decision Tree, Random Forest, Naive Bayes, etc.
Algorithms available for classification include Logistic regression, KNN Regression, Decision Tree, Random Forest, Naive Bayes, Support Vector Machine, Ada Boost, etc.
Provides evaluation metrics such as Mean Square Error, Mean Absolute Error, R2 Score for regression, and precision, recall, F1 score for classification.
### Download:

Enables users to download cleaned or reduced data and trained models in the form of pickle files.
Allows prediction and classification for new data points using selected algorithms.

### _Steps to Execute:_

1 Visit the web application at the following link:  https://automl-77tgcts4pkqowntarw5tbg.streamlit.app/

2 Choose the dataset file in either .xlsx or .csv format.

3 Upload the dataset and configure parameters as per your requirements.

4 The system will process the data in the background using Machine Learning algorithms.

5 Follow on-screen instructions to navigate through the various steps of the Auto ML process.


### Clone the Project:

To clone the project and run it locally, follow these steps:

1.Clone the repository from GitHub  :
    git clone <repository_url>
```bash
 git clone https://github.com/Ashwanth12/Auto_Ml.git
```
2.Navigate to the project directory:
```bash
    cd Auto-ML
```
 
3.Install dependencies:
  ```bash
    pip install -r requirements.txt
```  
4.Run the application:
```bash
    streamlit run main.py
```
    
