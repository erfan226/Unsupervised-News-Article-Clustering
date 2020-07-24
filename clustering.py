from language_processor import NLP
from utils import Utils
from preprocessor import Preprocessor
from constants import KMEANS_MODEL
import numpy as np
import math
from scipy.stats import multivariate_normal
import sys  # Remove later
np.set_printoptions(threshold=sys.maxsize)

utils = Utils()
prep = Preprocessor()
nlp = NLP()


class KMeans:
    def __init__(self, n_clusters, n_iterations=100):
        """
        K-Means clustering algorithm.

        :param int n_clusters: Number of clusters
        :param int n_iterations: Number of iterations to repeat for algorithm to find the best parameters. Will automatically exit if converged sooner
        """
        self.k = n_clusters
        self.iterations = n_iterations

    def init_centroids(self, n):
        """
        Initialize new k random centroids with numpy.

        :param int n: Number of dimensions which must match to that of documents
        :return: Initial values for k centroids
        """
        centroids = []
        for i in range(self.k):
            centroids.append(np.random.rand(n))
        return centroids

    def update_centroids(self, mu, clusters, doc_vectors):
        """
        Update previous centroids to new values based on clusters.

        :param list mu: Previous centroids to be updated
        :param dict clusters: Includes distance of data-point from centroid, closest centroid and document ID
        :param list doc_vectors: Vector of documents to measure the mean for new centroids
        :return: A list of k new calculated centroids
        """
        for cluster in clusters:
            cls_docs = []
            for data in clusters[cluster]:
                doc = doc_vectors[data[1]]
                cls_docs.append(doc)
            if len(cls_docs) != 0:
                mu[cluster] = [np.mean(k, dtype='float64') for k in zip(*cls_docs)]
        return mu

    def fit(self, doc_vectors):
        """
        Train KMeans algorithm based on initial centroids, updates until convergence, and clusters documents.

        :param list doc_vectors: Vector of documents
        :return:
        """
        print("Started KMeans...")
        mu = [doc_vectors[np.random.randint(len(doc_vectors)-1)], doc_vectors[np.random.randint(len(doc_vectors)-1)],
              doc_vectors[np.random.randint(len(doc_vectors)-1)], doc_vectors[np.random.randint(len(doc_vectors)-1)]]
        clusters = {}
        for iter_ in range(self.iterations):
            iterations = {0: {}, 1: {}, 2: {}, 3: {}}
            cls = [[], [], [], []]
            n_docs = len(doc_vectors)
            for d, doc_vector in enumerate(doc_vectors):
                clustered_doc = []
                for i in range(self.k):
                    dist = 0
                    for j, data_point in enumerate(doc_vector):
                        dist += np.linalg.norm(data_point - mu[i][j])
                    dist = math.sqrt(dist)
                    clustered_doc.append((dist, i))
                cluster_info = min(clustered_doc)
                cls[cluster_info[1]].append((cluster_info, doc_vectors.index(doc_vector)))
            iterations.update({0: cls[0], 1: cls[1], 2: cls[2], 3: cls[3]})
            clusters[iter_] = iterations
            mu = self.update_centroids(mu, clusters[iter_], doc_vectors)
            result = self.is_converged(iter_, clusters, mu)
            if result is not None or iter_ >= self.iterations-1:
                return self.predicted_cluster(result[0], n_docs)
            print("Iteration", iter_, "finished.")

    def is_converged(self, iter_, clusters, mu):
        """
        Checks if algorithm has reached convergence; then saves model parameters & returns clustered documents.

        :param int iter_: Current iteration of algorithm
        :param dic clusters: Cluster of documents
        :param list mu: Centroids of algorithm
        :return: Cluster of documents & the final centroids
        """
        if iter_ >= 2:
            if clusters[iter_] == clusters[iter_ - 2] or iter_ >= self.iterations-1:
                print("Reached the end of iteration!\n") if iter_ >= self.iterations - 1 else print("Converged!\n")
                utils.save_model({"centroids": mu}, KMEANS_MODEL, "json")
                return clusters[len(clusters) - 1], mu

    def predicted_cluster(self, clusters, n_docs):
        """
        Extracts documents indices from predicted result in training phase and passes the data to the evaluation function.

        :param dict clusters: Cluster of documents from training phase
        :param int n_docs: Number of given documents. Will be used to evaluate each cluster
        :return: List of clusters with documents indices
        """
        pred_labels = []
        for item in clusters:
            temp = []
            for i in clusters[item]:
                temp.append(i[1])
            pred_labels.append(temp)
        print("Clusters:", pred_labels, "\n")
        self.evaluate_clusters(pred_labels, n_docs)

    def evaluate_clusters(self, clusters, n_docs):
        """
        Evaluates each cluster, specifying major category & giving accuracy of each cluster.

        :param list clusters: List of k clusters including index of clustered documents
        :param int n_docs: Number of documents
        :return:
        """
        categories = {1: "Economics", 2: "Politics", 3: "Sports", 4: "Cultural"}
        n = n_docs
        cat1 = int((n/self.k)-1)
        cat2 = int((n/self.k) + cat1)
        cat3 = int((n/self.k) + cat2)
        cat4 = int((n/self.k) + cat3)

        for i, cluster in enumerate(clusters):
            cls = []
            c1, c2, c3, c4 = 0, 0, 0, 0
            cls_size = len(cluster)
            for item in cluster:
                if 0 <= item <= (n/self.k)-1:
                    c1 += 1
                elif cat1+1 <= item <= cat2:
                    c2 += 1
                elif cat2+1 <= item <= cat3:
                    c3 += 1
                elif cat3+1 <= item <= cat4:
                    c4 += 1
            if cls_size > 0:
                cls = [c1, c2, c3, c4]
                major_cat = max(cls)
                precision = major_cat/cls_size
                print("Cluster ", i + 1, " => ", cls, "\nMajor category: ", categories[cls.index(major_cat) + 1],
                      "\nPrecision: ", "%.2f" % round(precision * 100, 2), "%\n", sep="")
            else:
                print("Cluster", i + 1, "=>", "No documents was assigned.")

    def predict(self, doc_path):
        """
        Predicts the cluster that the new given document belongs to.

        :param str doc_path: Name of test file to predict & cluster
        :return:
        """
        test_data, df = prep.prep_test_doc(doc_path)
        mu = utils.load_model(KMEANS_MODEL, "json")
        tf_idf_vector = nlp.test_tf_idf(test_data, df)
        clustered_doc = []
        for i in range(self.k):
            dist = 0
            for j, data_point in enumerate(tf_idf_vector[0]):
                dist += np.linalg.norm(data_point - mu["centroids"][i][j])
            dist = math.sqrt(dist)
            clustered_doc.append((dist, i))
        prediction = min(clustered_doc)
        print("K-Means: Test data belongs to cluster", prediction[1]+1)


class GaussianMixture:
    def __init__(self, n_clusters, max_iter):
        """
        Mixture model algorithm.

        :param int n_clusters: Number of clusters
        :param int max_iter: Number of iterations to repeat for algorithm to find the optimized set of parameters
        """
        self.k = n_clusters
        self.max_iter = int(max_iter)
        self.shape = None
        self.row, self.col = None, None
        self.phi = None
        self.weights = None
        self.mu = None
        self.sigma = None

    def initialize(self, vectors):
        """
        Initialize the mean (mu_k), the covariance matrix (sigma_k) and the mixing coefficients (pi_k) by some (random)
        values.

        :param numpy.ndarray vectors: Vectors of documents
        :return:
        """
        self.shape = np.shape(vectors)
        self.row, self.col = self.shape

        self.phi = np.full(shape=self.k, fill_value=1 / self.k)
        self.weights = np.full(shape=self.shape, fill_value=1 / self.k)

        random_row = np.random.randint(low=0, high=self.row, size=self.k)

        self.mu = [vectors[row_index, :] for row_index in random_row]
        self.sigma = [np.cov(vectors.T) for _ in range(self.k)]

    def fit(self, vectors, dimensions=None):
        """
        Trains mixture model by estimating optimized parameters for a number of iterations & saving the results.
        Then clusters the given documents.

        :param numpy.ndarray vectors: Vectors of documents
        :return:
        """
        print("Started Mixture Models...")

        if dimensions is not None:
            transformed_vectors = utils.svd_transform(vectors, dimensions)
        else:
            transformed_vectors = np.array(vectors)

        self.initialize(transformed_vectors)
        for iteration in range(self.max_iter):
            print("Iteration:", iteration)
            self.e_step(transformed_vectors)
            self.m_step(transformed_vectors)
        print("Finished training.\nSaving parameters...\n")
        mu = [np.ndarray.tolist(i) for i in self.mu]
        sigma = [np.ndarray.tolist(i) for i in self.sigma]
        trained_model = {"means": mu, "sigma": sigma, "weights": np.ndarray.tolist(self.weights)}
        utils.save_model(trained_model, "mixModels_params", "json")
        self.cluster_data(transformed_vectors)

    def e_step(self, vectors):
        """
        Expectation-Step: Update weights & phi while mu & sigma (cov) stay constant
        :param numpy.ndarray vectors: Vectors of documents
        :return:
        """
        self.weights = self.parameter_estimation(vectors)
        self.phi = self.weights.mean(axis=0)

    def m_step(self, vectors):
        """
        M-Step: Update mu & sigma (cov) while phi & weights stay constant

        :param numpy.ndarray vectors: Vectors of documents
        :return:
        """
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (vectors * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(vectors.T, aweights=(weight / total_weight).flatten(), bias=True)

    def parameter_estimation(self, vectors):
        """
        Estimates weights of each cluster from the probability distribution.

        :param numpy.ndarray vectors: Vectors of documents
        :return: A list of weights for each cluster
        """
        likelihood = np.zeros((self.row, self.k))

        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i], allow_singular=True)
            likelihood[:, i] = distribution.pdf(vectors)

        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def cluster_data(self, vectors):
        """
        Clusters documents given data in parameter estimation (fit) phase.
        Only returns the most probable cluster for each document.

        :param numpy.ndarray vectors: Vectors of documents
        :return:
        """
        weights = self.parameter_estimation(vectors)
        clustered_docs = np.argmax(weights, axis=1)
        self.predicted_cluster(clustered_docs)

    def predicted_cluster(self, clustered_docs):
        """
        Separates documents in clusters with their index. Then evaluates them.

        :param numpy.ndarray clustered_docs: List of ordered documents with number of their predicted cluster
        :return:
        """
        clusters = [[], [], [], []]
        n_docs = len(clustered_docs.tolist())
        for doc_index, item in enumerate(clustered_docs.tolist()):
            clusters[item].append(doc_index)
        print("Clusters:", clusters, "\n")
        self.evaluate_clusters(clusters, n_docs)

    def evaluate_clusters(self, clusters, n_docs):
        """
        Evaluates each cluster, specifying major category & giving accuracy of each cluster.

        :param list clusters: List of k clusters including index of clustered documents
        :param int n_docs: Number of documents
        :return:
        """
        categories = {1: "Economics", 2: "Politics", 3: "Sports", 4: "Cultural"}
        n = n_docs
        cat1 = int((n/self.k)-1)
        cat2 = int((n/self.k) + cat1)
        cat3 = int((n/self.k) + cat2)
        cat4 = int((n/self.k) + cat3)

        for i, cluster in enumerate(clusters):
            cls = []
            c1, c2, c3, c4 = 0, 0, 0, 0
            cls_size = len(cluster)
            for item in cluster:
                if 0 <= item <= (n/self.k)-1:
                    c1 += 1
                elif cat1+1 <= item <= cat2:
                    c2 += 1
                elif cat2+1 <= item <= cat3:
                    c3 += 1
                elif cat3+1 <= item <= cat4:
                    c4 += 1
            if cls_size > 0:
                cls = [c1, c2, c3, c4]
                major_cat = max(cls)
                precision = major_cat/cls_size
                print("Cluster ", i + 1, " => ", cls, "\nMajor category: ", categories[cls.index(major_cat) + 1],
                      "\nPrecision: ", "%.2f" % round(precision * 100, 2), "%\n", sep="")
            else:
                print("Cluster", i + 1, "=>", "No documents was assigned.")

    def predict(self, doc_path, dimension_reduction=True):
        """
        Predicts the cluster that the given document belongs to.

        :param str doc_path: Name of test file to predict & cluster
        :param bool dimension_reduction: Use SVD model if training data was transformed as well
        :raises ValueError: May raise if some dimensions are not matched
        :return:
        """
        prediction = []

        test_data, df = prep.prep_test_doc(doc_path)

        if df is not None:
            test_vector = nlp.test_tf_idf(test_data, df)
            if dimension_reduction:
                svd = utils.load_model("svd_params", "pkl")
                transformed_vectors = svd.transform(test_vector)
            else:
                transformed_vectors = test_vector
        else:
            print("Error occurred! Check df model before continuing.")
            return

        params = utils.load_model("mixModels_params", "json")
        try:
            for m, c in zip(params["means"], params["sigma"]):
                prediction.append(
                    multivariate_normal(mean=m, cov=c, allow_singular=True).pdf(transformed_vectors) / np.sum(
                        [multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(transformed_vectors) for
                         mean, cov in zip(params["means"], params["sigma"])]))
        except ValueError as e:
            print("Error:", e, "\nMake sure dimension reduction option is not checked to False.")
            return
        self.prediction_result(prediction)

    def prediction_result(self, prediction):
        """
        Shows the result of predicting cluster of a new document.

        :param list prediction: Probability cluster of document(s)
        :return:
        """
        major_cluster = max(prediction)
        print("\nMixture Models. Tested new document:")
        for i, cls in enumerate(prediction):
            print("\nProbability of Cluster", i+1, "=>", cls, end="")
            if prediction.index(major_cluster) == i:
                print(" <= Major cluster!", end="")
