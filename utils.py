import json
import pickle
import julia
import numpy as np
from sklearn.decomposition import TruncatedSVD
from constants import DATA_DIR, TXT_EXT, DF_MODEL, STOPS, TEST


class Utils:
    def crawl_data(self, script_name):
        """
        Runs a Julia script to crawl data.

        :param str script_name: Name of Julia script
        :return:
        """
        jul = julia.Julia(compiled_modules=False)
        jul.include(script_name)

    def file_reader(self, path, limit=None):
        """
        Read the data file and converts to a list.

        :param str path: Path to the document
        :param int limit: Limit each document to a number of words
        :return: List of documents
        """
        data = []
        try:
            with open(path) as file:
                if limit:
                    for item in file:
                        item = item.strip().split()[:limit]
                        item = " ".join(item)
                        data.append(item)
                else:
                    for item in file:
                        data.append(item.strip())
            return data
        except FileNotFoundError:
            print("File not found at", path)
            exit(1)

    def sep_data(self, data):
        """
        Separates parts of news data.

        :param list data: String of documents
        :return: Dictionary of document with different parts of news separated
        """
        dic = {}
        sep_data = []
        for news in data:
            sep_data.append(news.split(","))
            for i, item in enumerate(sep_data):
                dic[i] = {}
                dic[i]["عنوان"] = item[0]
                dic[i]["لید"] = item[1]
                dic[i]["متن"] = item[2]
        return dic

    def merge_data(self, n_docs, limit, *paths):
        """
        Merges the data from different files and returns them as a list.

        :param int n_docs: Number of items to equally divide the data to. (Ex: if k=20, for 4 data-set it'll be a total of 80)
        :param int limit: Limit each document to a number of words
        :param str paths: Path of files to be merged. Only file name is needed (without directory or extension)
        :return: List of merged data in the given order
        """
        data = []
        for path in paths:
            new_data = self.file_reader(DATA_DIR+path+TXT_EXT, limit)[:n_docs]
            data.extend(new_data)
        return data

    def save_file(self, data, fn):
        """
        Saves the given data to the specified file.

        :param list data: Data to be saved
        :param str fn: Name of the file to be saved
        :return:
        """
        file = open(DATA_DIR+fn+TXT_EXT, "w")
        file.writelines(data)
        file.close()

    def save_model(self, data, fn, type_):
        """
        Takes in a model's parameters and saves it as a JSON/Binary file.

        :param data: Trained model parameters to be saved
        :param str fn: FileName
        :param str type_: Type of the file to be saved (json or pkl)
        """
        if type_ == "json":
            with open(DATA_DIR + "/models/" + fn + "." + type_, "w") as f:
                json.dump(data, f)
        elif type_ == "pkl":
            pkl_file = open(DATA_DIR + "/models/" + fn + "." + type_, 'wb')
            pickle.dump(data, pkl_file)
            pkl_file.close()
        else:
            print("Chosen type is not available!")

    def load_model(self, fn, type_):
        """
        Opens and returns a model's parameters.

        :param str fn: FileName
        :param str type_: Type of model (json or pkl)
        :return: Trained model parameters
        """
        try:
            if type_ == "json":
                with open(DATA_DIR + "/models/" + fn + ".json") as f:
                    model = json.load(f)
                    return model
            elif type_ == "pkl":
                pkl_file = open(DATA_DIR + "/models/" + fn + ".pkl", 'rb')
                model = pickle.load(pkl_file)
                return model
            else:
                print("Chosen type is not available!")
        except FileNotFoundError:
            print("Model", fn, "not found!")

    def svd_transform(self, vectors, n_components):
        """
        Fits SVD model on documents, saves its parameters and transforms those documents to a reduced dimension.

        :param list vectors: Vectors of documents
        :param int n_components: Number of dimension to reduce features of vectors to. Can not be more than the number of features or documents
        :return: Vectors with reduced dimensions
        """
        svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
        svd_fit = svd.fit(np.array(vectors))
        self.save_model(svd_fit, "svd_params", "pkl")
        transformed_vectors = svd.transform(vectors)
        return transformed_vectors
