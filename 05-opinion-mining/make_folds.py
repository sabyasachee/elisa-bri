# author Sabyasachee Baruah

import numpy as np
import pandas as pd
import os
import json
import config

def make_folds(n_folds):
    mpqa3_doc_ids = sorted(open(os.path.join(config.MPQA3_FOLDER, "doclist")).read().strip().split("\n"))
    mpqa3_docs = []

    for doc_id in mpqa3_doc_ids:
        doc_file = os.path.join(config.MPQA3_PROCESSED_FOLDER, doc_id, "tokenized.json")
        doc = json.load(open(doc_file))
        mpqa3_docs.append(doc)

    fold_size = int(len(mpqa3_docs)/n_folds)
    index = np.arange(len(mpqa3_docs))
    
    np.random.seed(2)
    np.random.shuffle(index)
    n_tuples_per_fold = np.zeros(n_folds)
    n_sentences_per_fold = np.zeros(n_folds, dtype=int)
    fold_records = []

    for i in range(n_folds):
        fold_docs = [mpqa3_docs[index[j]] for j in range(i * fold_size, (i + 1) * fold_size)]
        fold_records.extend([[mpqa3_doc_ids[index[j]], i] for j in range(i * fold_size, (i + 1) * fold_size)])
        for doc in fold_docs:
            n_sentences_per_fold[i] += len(doc)
            n_tuples_per_fold[i] += sum(len(sentence["dse"]) for sentence in doc)
    
    print("number of tuples per fold = {}, std. dev = {:.2f}".format(n_tuples_per_fold, np.std(n_tuples_per_fold)))
    print("number of sents  per fold = {}, std. dev = {:.2f}".format(n_sentences_per_fold, np.std(n_sentences_per_fold)))

    os.makedirs(os.path.join(config.RESULTS_FOLDER, "mpqa3"), exist_ok=True)
    fold_info_df = pd.DataFrame(fold_records, columns=["doc_id", "fold"])
    fold_info_df.to_csv(os.path.join(config.RESULTS_FOLDER, "mpqa3", "{}fold.csv".format(n_folds)), index=False)

if __name__ == "__main__":
    make_folds(2)