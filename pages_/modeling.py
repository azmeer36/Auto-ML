from pages_.imports import*

from models.supervised.regression import RegressionModel
from models.supervised.classification import ClassificationModel
from models.unsupervised.clustering import ClusteringModel
from models.unsupervised.dimen_red import DimensionalityReductionModel


def show():
    df_results = []
    df = pd.read_csv("./data.csv")
    df = df.iloc[:, :]
    df_clone = df.iloc[:, :]
    tab1, tab2 = st.tabs(["Supervised Learning", "Unsupervised Learning"])
    with tab1:
        st.subheader("Supervised Machine Learning")
        choice1 = st.selectbox("Task", ["Regression", "Classification"])
        chosen_target = st.selectbox('Choose the Target Column', df.columns, index=len(df.columns) - 1)
        st.session_state['chosen_target'] = chosen_target
        test_size = st.slider('Test Size', 0.01, 0.5, value=0.2)  # Default value set to 0.2
        # Calculate the training size
        train_size = 1 - test_size
        # Display the train/test split
        st.write(f"Training Size: {train_size:.2f} | Test Size: {test_size:.2f}")
        random_state = st.slider('Random_State', 0, 100)
        X = df.drop(columns=[chosen_target])
        y = df[chosen_target]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        numerical_col = X.select_dtypes(include=np.number).columns
        st.session_state['numerical_col_set'] = set(numerical_col)
        categorical_col = X.select_dtypes(exclude=np.number).columns
        st.session_state['categorical_col_set'] = set(categorical_col)
        scaler = MinMaxScaler()
        ct_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_col)],
                                        remainder='passthrough')
        st.session_state['ct_encoder'] = ct_encoder
        x_train_encoded = ct_encoder.fit_transform(x_train)
        x_test_encoded = ct_encoder.transform(x_test)
        x_train = scaler.fit_transform(x_train_encoded)
        x_test = scaler.transform(x_test_encoded)

        if "Regression" in choice1:    
            algorithm = st.selectbox("Regression Algorithms",
                                        ["Linear Regression", "Polynomial Regression",
                                        "Support Vector Regression", "Decision Tree Regression",
                                        "Random Forest Regression", "Ridge Regression",
                                        "Lasso Regression", "Gaussian Regression", "KNN Regression", "AdaBoost"])
            
            regression_model = RegressionModel(x_train, y_train, x_test, y_test)
            
            # Prepare parameters for specific algorithms
            params = {}
            if algorithm == "Polynomial Regression":
                params['degree'] = st.slider("Polynomial Degree", 2, 10, 2)
            if algorithm == "Support Vector Regression":
                params['kernel'] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                params['epsilon'] = st.slider("Svm Epsilon", 0.01, 1.0, 0.1)
            if algorithm == "Random Forest Regression":
                params['n_estimators'] = st.slider("Number of Estimators for Random Forest", 10, 200, 100)
            if algorithm == "Ridge Regression":
                params['alpha'] = st.slider("Ridge Alpha", 0.01, 10.0, 1.0)
            if algorithm == "Lasso Regression" :
                params['alpha'] = st.slider("Lasso Alpha", 0.01, 10.0, 1.0)
            if algorithm == "KNN Regression" :
                params['k_neighbors'] = st.slider("Number of Neighbors (K) for KNN", 1, 20, 5)
            if algorithm == "AdaBoost" :
                params['n_estimators'] = st.slider("Number of Estimators for AdaBoost", 10, 200, 100)


            if st.button('Run Modelling'):
                regression_model.run_selected_algorithms(algorithm, **params)
                df_results = regression_model.get_results()
                st.dataframe(df_results)
            
            
        elif "Classification" in choice1:     
            label = {}
            classes = {}
            v = 0
            for i in y.unique():
                label[i] = v
                classes[v] = i
                v += 1
            st.session_state['classes'] = classes
            y_test = y_test.apply(lambda x: label[x])
            y_train = y_train.apply(lambda x: label[x])

            algorithm = st.selectbox(
                "Classification Algorithms",
                ["Logistic Regression", "Decision Trees", "Random Forest", "Naive Bayes",
                "Support Vector Machines (SVM)", "Gradient Boosting", "Neural Networks",
                "Quadratic Discriminant Analysis (QDA)", "Adaptive Boosting (AdaBoost)",
                "Gaussian Processes", "Perceptron", "KNN Classifier", "Ridge Classifier",
                "Passive Aggressive Classifier", "Elastic Net", "Lasso Regression"])

            if st.button('Run Modelling'):
                classification_model = ClassificationModel(x_train_encoded, y_train, x_test, y_test)

                k_neighbors = None
                if algorithm == "KNN Classifier":
                    k_neighbors = st.slider("Number of Neighbors for KNN", 1, 20, 5)

                classification_model.run_selected_algorithms(algorithm, k_neighbors)

                df_results = classification_model.get_results()
                st.dataframe(df_results)

    with tab2:
        st.subheader("Unsupervised Machine Learning")
        numerical_col = df.select_dtypes(include=np.number).columns
        categorical_col = df.select_dtypes(exclude=np.number).columns
        ct_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_col)],
                                        remainder='passthrough')
        df = ct_encoder.fit_transform(df)
        choice1 = st.selectbox("Unsupervised", ["Clustering", "Dimensionality Reduction"])
        
        if "Clustering" in choice1:  
            algorithm = st.selectbox("Clustering Algorithms",
                                        ["Affinity Propagation", "Agglomerative Clustering",
                                        "BIRCH", "DBSCAN", "K-Means", "Mini-Batch K-Means",
                                        "Mean Shift", "OPTICS", "Spectral Clustering",
                                        "Gaussian Mixture Model"])
            if st.button('Run Model'):
                clustering_model = ClusteringModel(df)
                
                clustering_model.run_selected_algorithms(algorithm)

                df_results = clustering_model.get_results()
                st.dataframe(df_results)


        elif "Dimensionality Reduction" in choice1:
            scaler = StandardScaler()
            X_train_std = scaler.fit_transform(x_train)
            X_test_std = scaler.transform(x_test)

            algorithms = st.selectbox("Dimensionality Reduction",
                                    ["PCA", "LDA", "Truncated SVD", "t-SNE", "MDS", "Isomap"])
            nc = st.slider("n_components", 1, df_clone.shape[1])

            if st.button('Run Model'):
                reduction_model = DimensionalityReductionModel(X_train_std, y_train, X_test_std, df_clone)

                df_results, result_message = reduction_model.run_reduction(algorithms, nc)
                st.write(result_message)
                st.dataframe(df_results)
            