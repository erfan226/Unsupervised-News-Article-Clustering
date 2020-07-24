from __future__ import unicode_literals
import re
from hazm import *
from utils import Utils
from constants import DATA_DIR, TXT_EXT, DF_MODEL, STOPS

utils = Utils()


class Preprocessor:
    """
    Preprocessor Core.
    """
    def normalizer(self, data: list):
        """
        Normalize the data with Hazm library.

        :param list data: Data to be normalized
        :return: Documents with normalized characters
        """
        normalizer = Normalizer()
        normalized_data = []
        for item in data:
            normalized_item = normalizer.normalize(item)
            normalized_data.append(normalized_item)
        return normalized_data

    def process_raw_data(self, stops, tag, *paths):
        """
        Same as data_preprocessor but just reads multiple documents together and passes them to data_preprocessor.

        :param str stops: Path to stop words file
        :param str tag: Part to extract (title, lead or body)
        :param str paths: Path to raw news data
        :return:
        """
        for path in paths:
            self.data_preprocessor(path, stops, tag)

    def data_preprocessor(self, path, stops, capture, lemmatize = False):
        """
        Reads raw data from the given path and pre-process the selected parts and saves it in a new file

        :param str path: Path to raw news data
        :param str stops: Path to stop words file
        :param str capture: Part to extract (title, lead or body)
        :return:
        """
        raw_data = utils.file_reader("data/raw/"+path+".txt")
        stops = utils.file_reader("data/"+stops+".txt")
        news_data = utils.sep_data(raw_data)

        title = []
        lead = []
        body = []
        normal_news = ""
        for item in news_data:
            if capture == "تیتر":
                title.append(news_data[item][capture])
                normal_news = self.normalizer(title)
            elif capture == "لید":
                lead.append(news_data[item][capture])
                normal_news = self.normalizer(lead)
            elif capture == "متن":
                body.append(news_data[item][capture])
                normal_news = self.normalizer(body)
            else:
                print(capture+" صحیح نیست! فقط تیتر، لید یا متن مجاز می‌باشد.")
                return
        tokenized_news = self.tokenizer(normal_news)
        filtered_news = self.remove_stop_words(tokenized_news, stops, lemmatize)

        news_data = []
        for i, item in enumerate(filtered_news):
            news_data.append(" ".join(filtered_news[i]) + "\n")

        utils.save_file(news_data, path)

    def tokenizer(self, data: list):
        """
        Tokenize the data with Hazm library

        :param list data: Data to be tokenized
        :return: Tokenized list of documents
        """
        tokenized_list = []
        for item in data:
            tokens = word_tokenize(item)
            tokenized_list.append(tokens)
        return tokenized_list

    def remove_stop_words(self, data: list, stops: list, lemmatize: bool):
        """
        Removes stopwords and filters the data.

        :param list data: Data to be cleaned
        :param list stops: A list of stopwords to check on with and remove them from data
        :param bool lemmatize: If True, will also lemmatize tokens with Hazm library. Better not to use!
        :return: Cleaned data
        """
        output = []
        pattern = '[^\u0621-\u06CC#\r]'
        for tokens in data:
            filtered_words = []
            if lemmatize:
                # todo: Convert to function and fix if broken
                lemmatizer = Lemmatizer()
                stemmer = Stemmer()
                for token in tokens:
                    if token not in stops and len(re.findall(pattern, token)) <= 0:
                        if token.startswith("#"):
                            filtered_words.append(token)
                        else:
                            token = lemmatizer.lemmatize(token)
                            if token.find("#") != -1 and not token.startswith("#"):
                                filtered_words.append(token.split('#')[1])
                            else:
                                filtered_words.append(token)
                output.append(filtered_words)
            else:
                for token in tokens:
                    if token not in stops and len(re.findall(pattern, token)) <= 0:
                        filtered_words.append(token)
                output.append(filtered_words)
        return output

    def prep_test_doc(self, doc_path):
        """
        Reads a test document from /data/raw & pre-processes it.

        :param str doc_path: Name of test file to load & convert
        :return: Processed data & document frequency counts
        """
        self.data_preprocessor(doc_path, STOPS, "متن")
        test_data = utils.file_reader(DATA_DIR + doc_path + TXT_EXT)
        df = utils.load_model(DF_MODEL, "json")
        return test_data, df
