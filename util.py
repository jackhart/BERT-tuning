import numpy as np
import pandas as pd
import string


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def clean_text(text, stem=False, string_r=False):
    """
    Cleans tweet for tf-idf vectorizer
    :param text: string; tweet to clean
    """
    stops = set(stopwords.words('english'))
    toks = word_tokenize(text)  # .lower()
    if stem:
        stemmer = PorterStemmer()
        toks = [stemmer.stem(tok) for tok in toks]

    toks_nopunc = [tok for tok in toks if tok not in string.punctuation]
    toks_nostop = [tok for tok in toks_nopunc if tok not in stops]

    if string_r:
        return " ".join(toks_nostop)

    return toks_nostop


def load_tweets(file, stem=False, string_r=False):
    """
    loads in Twitter dataset and cleans it
    :param file: string; file to dataset
    :param stem: bool; optional, whether to stem each token
    :param string_r: bool, optional, whether to return full string or list
    """
    data_set = pd.read_csv(file)
    corpus = data_set.text.to_numpy()
    corpus = [line.replace('\r', '').replace('\n', '') for line in corpus.tolist()]
    classes = data_set.target.tolist()

    # clean corpus
    corpus_cleaned = [clean_text(text, string_r=string_r, stem=stem) for text in corpus]

    return corpus_cleaned, classes


