from models.imports import *
import pickle

class RegressionModel:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.table = {"Algorithm": [], "MAE": [], "RMSE": [], "R2 Score": []}

    def evaluate_model(self, model, model_name):
        y_pred = model.predict(self.x_test)
        mse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2score = r2_score(self.y_test, y_pred) * 100

        self.table["RMSE"].append(mse)
        self.table["MAE"].append(mae)
        self.table["Algorithm"].append(model_name)
        self.table["R2 Score"].append(r2score)

    def linear_regression(self):
        reg = LinearRegression()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Linear Regression")
        pickle.dump(reg, open('LR.pkl', 'wb'))

    def polynomial_regression(self, degree):
        reg = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, f"Polynomial Regression (Degree: {degree})")
        pickle.dump(reg, open('PR.pkl', 'wb'))

    def support_vector_regression(self, kernel, epsilon):
        reg = SVR(kernel=kernel, epsilon=epsilon)
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, f"Support Vector Regression (Kernel: {kernel}, Epsilon: {epsilon})")
        pickle.dump(reg, open('SVR.pkl', 'wb'))

    def decision_tree_regression(self):
        reg = DecisionTreeRegressor()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Decision Tree Regression")
        pickle.dump(reg, open('DTR.pkl', 'wb'))

    def random_forest_regression(self, n_estimators):
        reg = RandomForestRegressor(n_estimators=n_estimators)
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, f"Random Forest Regression (Estimators: {n_estimators})")
        pickle.dump(reg, open('RFR.pkl', 'wb'))

    def ridge_regression(self, alpha):
        reg = Ridge(alpha=alpha)
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, f"Ridge Regression (Alpha: {alpha})")
        pickle.dump(reg, open('RR.pkl', 'wb'))

    def lasso_regression(self, alpha):
        reg = Lasso(alpha=alpha)
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, f"Lasso Regression (Alpha: {alpha})")
        pickle.dump(reg, open('LASR.pkl', 'wb'))

    def gaussian_regression(self):
        reg = GaussianProcessRegressor()
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "Gaussian Regression")
        pickle.dump(reg, open('GR.pkl', 'wb'))

    def knn_regression(self, k_neighbors):
        reg = KNeighborsRegressor(n_neighbors=k_neighbors)
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, f"K-Nearest Neighbors Regression (K={k_neighbors})")
        pickle.dump(reg, open('KNR.pkl', 'wb'))

    def adaboost_regression(self, n_estimators):
        reg = AdaBoostRegressor(n_estimators=n_estimators)
        reg.fit(self.x_train, self.y_train)
        self.evaluate_model(reg, "AdaBoost")
        pickle.dump(reg, open('ABR.pkl', 'wb'))
        
    def run_selected_algorithms(self, algorithm, **kwargs):
        if algorithm == "Linear Regression":
            self.linear_regression()
        elif algorithm == "Polynomial Regression":
            self.polynomial_regression(kwargs.get('degree', 2))
        elif algorithm == "Support Vector Regression":
            self.support_vector_regression(kwargs.get('kernel', 'linear'), kwargs.get('epsilon', 0.1))
        elif algorithm == "Decision Tree Regression":
            self.decision_tree_regression()
        elif algorithm == "Random Forest Regression":
            self.random_forest_regression(kwargs.get('n_estimators', 100))
        elif algorithm == "Ridge Regression":
            self.ridge_regression(kwargs.get('alpha', 1.0))
        elif algorithm == "Lasso Regression":
            self.lasso_regression(kwargs.get('alpha', 1.0))
        elif algorithm == "Gaussian Regression":
            self.gaussian_regression()
        elif algorithm == "KNN Regression":
            self.knn_regression(kwargs.get('k_neighbors', 5))
        elif algorithm == "AdaBoost":
            self.adaboost_regression(kwargs.get('n_estimators', 100))

    def get_results(self):
        return pd.DataFrame(self.table)