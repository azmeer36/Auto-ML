from models.imports import *

class ClusteringModel:
    def __init__(self, df):
        self.df = df
        self.table = {"Algorithm": [], "Silhouette": []}

    def evaluate_model(self, model, model_name):
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(self.df)
        uniq = len(np.unique(labels))
        if uniq == 1:
            st.warning("Improper Data for " + model_name)
            return None
        silhouette_score = metrics.silhouette_score(self.df, labels) * 100
        self.table["Algorithm"].append(model_name)
        self.table["Silhouette"].append(silhouette_score)
        return model

    def affinity_propagation(self):
        model = AffinityPropagation()
        model.fit(self.df)
        return self.evaluate_model(model, "Affinity Propagation")

    def agglomerative_clustering(self):
        model = AgglomerativeClustering()
        model.fit(self.df)
        return self.evaluate_model(model, "Agglomerative Clustering")

    def birch(self):
        model = Birch()
        model.fit(self.df)
        return self.evaluate_model(model, "BIRCH")

    def dbscan(self):
        model = DBSCAN()
        model.fit(self.df)
        return self.evaluate_model(model, "DBSCAN")

    def k_means(self):
        model = KMeans()
        model.fit(self.df)
        return self.evaluate_model(model, "K-Means")

    def mini_batch_kmeans(self):
        model = MiniBatchKMeans()
        model.fit(self.df)
        return self.evaluate_model(model, "Mini-Batch K-Means")

    def mean_shift(self):
        model = MeanShift()
        model.fit(self.df)
        return self.evaluate_model(model, "Mean Shift")

    def optics(self):
        model = OPTICS()
        model.fit(self.df)
        return self.evaluate_model(model, "OPTICS")

    def spectral_clustering(self):
        model = SpectralClustering()
        model.fit(self.df)
        return self.evaluate_model(model, "Spectral Clustering")

    def gaussian_mixture(self):
        model = GaussianMixture(n_components=10)
        model.fit(self.df)
        return self.evaluate_model(model, "Gaussian Mixture Model")
    
    def run_selected_algorithms(self, algorithm):
        if algorithm == "Affinity Propagation":
            self.affinity_propagation()
        elif algorithm == "Agglomerative Clustering":
            self.agglomerative_clustering()
        elif algorithm == "BIRCH":
            self.birch()
        elif algorithm == "DBSCAN":
            self.dbscan()
        elif algorithm == "K-Means":
            self.k_means()
        elif algorithm == "Mini-Batch K-Means":
            self.mini_batch_kmeans()
        elif algorithm == "Mean Shift":
            self.mean_shift()
        elif algorithm == "OPTICS":
            self.optics()
        elif algorithm == "Spectral Clustering":
            self.spectral_clustering()
        elif algorithm == "Gaussian Mixture Model":
            self.gaussian_mixture()

    def get_results(self):
        return pd.DataFrame(self.table)