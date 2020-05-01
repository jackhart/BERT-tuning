import argparse
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

from util import load_tweets

# define data processors
data_processors = {'TfidfVectorizer': {'function': TfidfVectorizer,
                                       'kwargs': {"use_idf": True, "min_df": 10}},
                   'CountVectorizer': {'function': CountVectorizer,
                                       'kwargs': {}},
                   'HashingVectorizer': {'function': HashingVectorizer,
                                         'kwargs': {}}}


def train_test(feature_extractor, train_texts, test_texts,
                                  train_class, test_class):

    # build vectorizer object
    vectorizer = feature_extractor['function']("content", lowercase=False, analyzer="word",
                                               token_pattern="\S+", **feature_extractor['kwargs']).fit(train_texts)

    # transform training and test data
    tfidf_train = vectorizer.transform(train_texts)
    tfidf_test = vectorizer.transform(test_texts)

    del vectorizer

    # train the logisitc regression
    clf = LogisticRegression(random_state=88).fit(tfidf_train, train_class)

    # calculate predictions on dev set
    dev_pred = clf.predict(tfidf_test)

    # calculate metrics
    p = precision_score(test_class, dev_pred)
    r = recall_score(test_class, dev_pred)
    f = f1_score(test_class, dev_pred)
    a = accuracy_score(test_class, dev_pred)

    return p, r, f, a


def main(args):
    """Fits a logistic regression with different vectorized representations of the twitter data"""

    # get feature extractor
    feature_extractor = data_processors[args.feature_extractor]

    # load train and test
    train_texts, train_class = load_tweets(args.train_file, stem=True, string_r=True)
    test_texts, test_class = load_tweets(args.test_file, stem=True, string_r=True)

    if args.split_type == "holdout":
        p, r, f, a = train_test(feature_extractor, train_texts, test_texts, train_class, test_class)
        print(f"Scores:\nprecision {p:0.03}\nrecall {r:0.03}\nf1 {f:0.03}\naccuracy {a:0.03}")

    elif args.split_type == "cv":
        # combine train and test data, then split by 10
        combined_text = np.array(train_texts + test_texts)
        combined_class = np.array(train_class + test_class)

        kf = KFold(n_splits=10, random_state=args.seed, shuffle=True)
        train_test_indices = kf.split(combined_class)

        prec, recall, f1, acc = [], [], [], []
        for train_inx, test_inx in train_test_indices:

            train_texts, train_class = combined_text[train_inx].tolist(), combined_class[train_inx].tolist()
            test_texts, test_class = (combined_text[~np.isin(np.arange(len(combined_text)), train_inx)].tolist(),
                                      combined_class[~np.isin(np.arange(len(combined_text)), train_inx)].tolist())

            p, r, f, a = train_test(feature_extractor, train_texts, test_texts, train_class, test_class)
            print("CV Trial:")
            print(f"Scores:\nprecision {p:0.03}\nrecall {r:0.03}\nf1 {f:0.03}\naccuracy {a:0.03}\n")

            # append lists
            prec.append(p)
            recall.append(r)
            f1.append(f)
            acc.append(a)

        if args.save_results:
            df = pd.DataFrame({'precision': prec, 'recall': recall,
                                'f1': f1, 'accuracy': acc})
            df.to_csv(args.save_results)

    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train.csv",
                        help="test file")
    parser.add_argument("--test_file", type=str, default="data/test.csv",
                        help="train file")

    parser.add_argument("--feature_extractor", type=str, default="TfidfVectorizer",
                        help="feature extractor type")
    parser.add_argument("--split_type", type=str, default="holdout",
                        help="either holdout or cv")
    parser.add_argument("--seed", type=int, default=57,
                        help="seed for split in 10-fold cv")
    parser.add_argument("--save_results", type=str, default=None,
                        help="indicates where results from cv should be saved."
                             "If not indicated, they results arn't saved.")

    args = parser.parse_args()

    main(args)