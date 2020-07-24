import math
from utils import Utils
from constants import ECONOMICS, POLITICS, SPORTS, CULTURE

utils = Utils()


class NLP:
    """
    Main module for NLP tasks.

    """
    def tokenizer(self, data, unique_only):
        """
        Tokenize the given documents. Can be used to extract feature vector.

        :param list data: Documents to be tokenized
        :param bool unique_only: Returns every token only once if true (i.e. the feature vector)
        :return: Tokens or feature vector
        """
        tokens = []
        for sentence in data:
            temp_tokens = []
            words = sentence.split()
            for word in words:
                temp_tokens.append(word)
            tokens.append(temp_tokens)
        if unique_only:
            unique_tokens = []
            for item in tokens:
                unique_tokens = list(set(unique_tokens + item))
            return unique_tokens
        else:
            return tokens

    # todo: Remove if unused
    def tf_count(self, tokens, doc_tokens):
        tf_counts = []
        for token in tokens:
            if token in doc_tokens:
                tf_counts.append((token, doc_tokens.count(token)))
        return tf_counts

    def df_counts(self, features, docs):
        """
        Computes document frequency for each token.

        :param list docs: Documents to extract tokens from
        :return: Dictionary of token document frequency and number of documents used for training
        """
        DF = {}
        number_of_docs = len(docs)
        for i in range(number_of_docs):
            doc = docs[i].split(" ")
            for token in features:
                if token != "" and token in doc:
                    try:
                        DF[token].add(i)
                    except:
                        DF[token] = {i}
        # Replacing the doc list with its count
        for i in DF:
            DF[i] = len(DF[i])
        DF["n_docs"] = number_of_docs
        utils.save_model(DF, "df_counts", "json")
        return DF

    def tf_idf(self, features, docs):
        """
        Convert test documents to tf-idf vectors.

        :param features:
        :param list docs: Documents to be converted
        :return: List of tf-idf vectors for given documents
        """
        df = self.df_counts(features, docs)
        tf_idf = []
        n_docs = len(docs)
        for i in range(n_docs):
            doc_tokens = docs[i].split(" ")
            n_doc_tokens = len(doc_tokens)
            temp_vec = []
            for token in df:
                if token != "n_docs":
                    tf = doc_tokens.count(token) / n_doc_tokens
                    # tf_idf[i, token] = round(tf * math.log(n_docs / (df[token] + 1), 2), 3) # Dictionary instead of list
                    temp_vec.append(round(tf * math.log(n_docs / (df[token] + 1), 2), 3))
            tf_idf.append(temp_vec)
        utils.save_model(tf_idf, "tfidf_vectors", "json")
        return tf_idf

    def test_tf_idf(self, docs, df):
        """
        Convert test documents to tf-idf vectors.

        :param list docs: Documents to be converted
        :param dict df: Count of document frequencies
        :return: List of TF-IDF vectors of given documents
        """
        tf_idf = []
        n_docs = len(docs)
        n_corpus_docs = df["n_docs"]

        for i in range(n_docs):
            doc_tokens = docs[i].split(" ")
            n_doc_tokens = len(doc_tokens)
            temp_vec = []
            for token in df:
                if token != "n_docs":
                    tf = doc_tokens.count(token) / n_doc_tokens
                    # tf_idf[i, token] = round(tf * math.log(n_docs / (df[token] + 1), 2), 3) # Dictionary instead of list
                    temp_vec.append(round(tf * math.log(n_corpus_docs / (df[token] + 1), 2), 3))
            tf_idf.append(temp_vec)
        return tf_idf

    def doc_to_vec(self, n_docs=250, trim_length=200, load_vectors=False):
        """
        Merges data, creates feature vector, and computes/loads tf-idf vectors.

        :param int n_docs: Number of documents to equally divide the data to. Only works if load_vectors is False (Ex: if k=20 * 4 = 80 docs)
        :param int trim_length: Limit each document to a number of words. Only works if load_vectors is False
        :param bool load_vectors: Loads computed vectors if true; Creates & saves vectors if not
        :return: TF-IDF vectors of given documents
        """
        if load_vectors:
            try:
                tf_idf_vector = utils.load_model("tfidf_vectors", "json")
            except FileNotFoundError:
                print("tfidf_vectors file does not exist!")
                return
        else:
            data = utils.merge_data(n_docs, trim_length, ECONOMICS, POLITICS, SPORTS, CULTURE)
            feature_vector = self.tokenizer(data, True)
            tf_idf_vector = self.tf_idf(feature_vector, data)
        return tf_idf_vector
