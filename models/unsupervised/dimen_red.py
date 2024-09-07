from models.imports import *

class DimensionalityReductionModel:
    def __init__(self, x_train, y_train, x_test, df):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.df = df

    def apply_pca(self, n_components):
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(self.df)
        return pd.DataFrame(data_pca), "PCA Results:"

    def apply_lda(self, n_components):
        lda = LDA(n_components=n_components)
        data_lda = lda.fit_transform(self.x_train, self.y_train)
        return pd.DataFrame(data_lda), "LDA Results:"

    def apply_truncated_svd(self, n_components):
        svd = TruncatedSVD(n_components=n_components)
        data_svd = svd.fit_transform(self.df)
        return pd.DataFrame(data_svd), "Truncated SVD Results:"

    def apply_tsne(self, n_components):
        tsne = TSNE(n_components=n_components, random_state=42)
        data_tsne = tsne.fit_transform(self.df)
        return pd.DataFrame(data_tsne), "t-SNE Results:"

    def apply_mds(self, n_components):
        mds = MDS(n_components=n_components, random_state=42)
        data_mds = mds.fit_transform(self.df)
        return pd.DataFrame(data_mds), "MDS Results:"

    def apply_isomap(self, n_components):
        isomap = Isomap(n_components=n_components, n_neighbors=5)
        data_isomap = isomap.fit_transform(self.df)
        return pd.DataFrame(data_isomap), "Isomap Results:"

    def run_reduction(self, algorithm, n_components):
        if algorithm == "PCA":
            return self.apply_pca(n_components)
        elif algorithm == "LDA":
            return self.apply_lda(n_components)
        elif algorithm == "Truncated SVD":
            return self.apply_truncated_svd(n_components)
        elif algorithm == "t-SNE":
            return self.apply_tsne(n_components)
        elif algorithm == "MDS":
            return self.apply_mds(n_components)
        elif algorithm == "Isomap":
            return self.apply_isomap(n_components)
        else:
            raise ValueError("Invalid algorithm selected.")