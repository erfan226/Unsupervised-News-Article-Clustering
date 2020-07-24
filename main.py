from utils import Utils
from preprocessor import Preprocessor
from language_processor import NLP
from clustering import KMeans, GaussianMixture
from constants import *

utils = Utils()
prep = Preprocessor()
nlp = NLP()

# Only uncomment if data crawling is needed again. Runs Julia script.
# utils.crawl_data("news_crawler.jl")

# Only uncomment if pre-processing raw data again, e.g. after crawling new data. Saves processed documents in /data/raw.
# prep.process_raw_data(STOPS, "متن", ECONOMICS, POLITICS, SPORTS, CULTURE)

# Creates tf-idf vectors from documents. Change to False if changing other arguments.
doc_vectors = nlp.doc_to_vec(250, 200, True)
err_msg = "Error occurred! Check tf_idf vectors before continuing."

k_means = KMeans(4, 15)
if doc_vectors is not None:
    # k_means.fit(doc_vectors)
    k_means.predict(TEST)
else:
    print(err_msg)

mix_model = GaussianMixture(4, 20)
if doc_vectors is not None:
    # mix_model.fit(doc_vectors, 30)
    mix_model.predict(TEST)
else:
    print(err_msg)
