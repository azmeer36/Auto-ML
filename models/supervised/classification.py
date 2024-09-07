from models.imports import *
import pickle

class ClassificationModel:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.table = {"Algorithm": [], "Precision": [], "Recall": [], "F1-Score": [], "Accuracy": []}

    def evaluate_model(self, model, model_name):
        y_pred = model.predict(self.x_test)
        precision = precision_score(self.y_test, y_pred, average='weighted') * 100
        recall = recall_score(self.y_test, y_pred, average='weighted') * 100
        f1 = f1_score(self.y_test, y_pred, average='weighted') * 100
        accuracy = accuracy_score(self.y_test, y_pred) * 100

        self.table["Algorithm"].append(model_name)
        self.table["Precision"].append(precision)
        self.table["Recall"].append(recall)
        self.table["F1-Score"].append(f1)
        self.table["Accuracy"].append(accuracy)

    def logistic_regression(self):
        reg = LogisticRegression()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Logistic Regression")
        pickle.dump(reg, open('LOR.pkl', 'wb'))

    def decision_tree(self):
        reg = DecisionTreeClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Decision Trees")
        pickle.dump(reg, open('DT.pkl', 'wb'))

    def random_forest(self):
        reg = RandomForestClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Random Forest")
        pickle.dump(reg, open('RF.pkl', 'wb'))

    def naive_bayes(self):
        reg = GaussianNB()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Naive Bayes")
        pickle.dump(reg, open('NB.pkl', 'wb'))

    def support_vector_machine(self):
        reg = SVC(decision_function_shape='ovo')
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Support Vector Machines (SVM)")
        pickle.dump(reg, open('SVM.pkl', 'wb'))

    def gradient_boosting(self):
        reg = GradientBoostingClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Gradient Boosting")
        pickle.dump(reg, open('GB.pkl', 'wb'))

    def neural_networks(self):
        reg = MLPClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Neural Networks")
        pickle.dump(reg, open('NN.pkl', 'wb'))

    def quadratic_discriminant_analysis(self):
        reg = QuadraticDiscriminantAnalysis()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Quadratic Discriminant Analysis (QDA)")
        pickle.dump(reg, open('QDA.pkl', 'wb'))

    def adaptive_boosting(self):
        reg = AdaBoostClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Adaptive Boosting (AdaBoost)")
        pickle.dump(reg, open('AB.pkl', 'wb'))

    def gaussian_processes(self):
        reg = GaussianProcessClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Gaussian Processes")
        pickle.dump(reg, open('GP.pkl', 'wb'))

    def perceptron(self):
        reg = Perceptron()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Perceptron")
        pickle.dump(reg, open('PT.pkl', 'wb'))

    def knn_classifier(self, k_neighbors):
        reg = KNeighborsClassifier(n_neighbors=k_neighbors)
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, f"K-Nearest Neighbors Classifier (K={k_neighbors})")
        pickle.dump(reg, open('KNC.pkl', 'wb'))

    def ridge_classifier(self):
        reg = RidgeClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Ridge Classifier")
        pickle.dump(reg, open('RC.pkl', 'wb'))

    def passive_aggressive_classifier(self):
        reg = PassiveAggressiveClassifier()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Passive Aggressive Classifier")
        pickle.dump(reg, open('PA.pkl', 'wb'))

    def elastic_net(self):
        x_train_df = pd.DataFrame(self.x_train)
        y_train_df = pd.DataFrame(self.y_train)
        x_test_df = pd.DataFrame(self.x_test)
        y_test_df = pd.DataFrame(self.y_test)

        x_train_df = x_train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train_df = y_train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        x_test_df = x_test_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_test_df = y_test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        reg = ElasticNet()
        reg.fit(x_train_df, y_train_df)
        y_pred = reg.predict(x_test_df)
        y_pred = pd.DataFrame(y_pred).apply(np.floor)

        self.evaluate_model(reg, "Elastic Net")
        pickle.dump(reg, open('EN.pkl', 'wb'))

    def lasso_regression(self):
        x_train_df = pd.DataFrame(self.x_train)
        y_train_df = pd.DataFrame(self.y_train)
        x_test_df = pd.DataFrame(self.x_test)
        y_test_df = pd.DataFrame(self.y_test)

        x_train_df = x_train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train_df = y_train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        x_test_df = x_test_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        y_test_df = y_test_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        reg = Lasso()
        reg.fit(x_train_df, y_train_df)
        y_pred = reg.predict(x_test_df)
        y_pred = pd.DataFrame(y_pred).apply(np.floor)

        self.evaluate_model(reg, "Lasso Regression")
        pickle.dump(reg, open('LAR.pkl', 'wb'))
        
    def run_selected_algorithms(self, algorithm, k_neighbors=None):
        if algorithm == "Logistic Regression":
            self.logistic_regression()
        elif algorithm == "Decision Trees":
            self.decision_tree()
        elif algorithm == "Random Forest":
            self.random_forest()
        elif algorithm == "Naive Bayes":
            self.naive_bayes()
        elif algorithm == "Support Vector Machines (SVM)":
            self.support_vector_machine()
        elif algorithm == "Gradient Boosting":
            self.gradient_boosting()
        elif algorithm == "Neural Networks":
            self.neural_networks()
        elif algorithm == "Quadratic Discriminant Analysis (QDA)":
            self.quadratic_discriminant_analysis()
        elif algorithm == "Adaptive Boosting (AdaBoost)":
            self.adaptive_boosting()
        elif algorithm == "Gaussian Processes":
            self.gaussian_processes()
        elif algorithm == "Perceptron":
            self.perceptron()
        elif algorithm == "KNN Classifier":
            if k_neighbors is not None:
                self.knn_classifier(k_neighbors)
        elif algorithm == "Ridge Classifier":
            self.ridge_classifier()
        elif algorithm == "Passive Aggressive Classifier":
            self.passive_aggressive_classifier()
        elif algorithm == "Elastic Net":
            self.elastic_net()
        elif algorithm == "Lasso Regression":
            self.lasso_regression()


    def get_results(self):
        return pd.DataFrame(self.table)