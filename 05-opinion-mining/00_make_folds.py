# author Sabyasachee Baruah
# 5 fold MPQA 3

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import config

def make_folds(mpqa3_folder, mpqa3_processed_folder, results_folder):
    mpqa3_doc_ids = sorted(open(os.path.join(mpqa3_folder, "doclist")).read().strip().split("\n"))
    mpqa3_docs = []

    for doc_id in mpqa3_doc_ids:
        doc_file = os.path.join(mpqa3_processed_folder, doc_id, "tokenized.json")
        doc = json.load(open(doc_file))
        mpqa3_docs.append(doc)

    n_folds = 5
    fold_size = int(len(mpqa3_docs)/n_folds)
    index = np.arange(len(mpqa3_docs))
    
    np.random.seed(15)
    np.random.shuffle(index)
    n_tuples_per_fold = np.zeros(n_folds)
    n_sentences_per_fold = np.zeros(n_folds, dtype=int)
    fold_records = []
    sentence_lens = []

    for i in range(n_folds):
        fold_docs = [mpqa3_docs[index[j]] for j in range(i * fold_size, (i + 1) * fold_size)]
        fold_records.extend([[mpqa3_doc_ids[index[j]], i] for j in range(i * fold_size, (i + 1) * fold_size)])
        for doc in fold_docs:
            n_sentences_per_fold[i] += len(doc)
            for sentence in doc:
                n_tuples_per_fold[i] += sum(1 for ex in sentence["dse"] if ex["target-type"] in ["entity", "event"])
                sentence_lens.append(len(sentence["tokens"]))
    
    print("number of tuples per fold = {}".format(n_tuples_per_fold))
    print("std. dev = {:.2f}".format(np.std(n_tuples_per_fold)))

    os.makedirs(os.path.join(results_folder, "mpqa3"), exist_ok=True)

    fold_info_df = pd.DataFrame(fold_records, columns=["doc_id", "fold"])
    fold_info_df.to_csv(os.path.join(results_folder, "mpqa3", "5fold.csv"), index=False)

    print("sentence lengths: mean = {:.2f}, median = {}, std = {:.2f}, max = {}".format(np.mean(sentence_lens), np.median(sentence_lens), np.std(sentence_lens), np.max(sentence_lens)))
    plt.figure(figsize = (10, 8))
    plt.hist(sentence_lens, bins=10)
    plt.savefig(os.path.join(results_folder, "mpqa3/sentence_lengths.png"))

    print("number of sentences per fold = {}".format(n_sentences_per_fold))

if __name__ == "__main__":
    make_folds(config.MPQA3_FOLDER, config.MPQA3_PROCESSED_FOLDER, config.RESULTS_FOLDER)