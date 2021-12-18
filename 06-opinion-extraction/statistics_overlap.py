# author : Sabyasachee Baruah

# Find frequency of overlap between spans of different labels

import os
import json
import numpy as np
import pandas as pd
from collections import Counter

import utils
import config

def statistics_overlap():
    mpqa2_filelist = open(os.path.join(config.RAW_FOLDER, "database.mpqa.2.0/doclist.attitudeSubset")).read()\
    .splitlines()
    mpqa3_filelist = open(os.path.join(config.RAW_FOLDER, "database.mpqa.3.0/doclist")).read().splitlines()
    
    mpqa2_overlap = np.zeros((5, 6))
    mpqa3_overlap = np.zeros((9, 10))

    mpqa2_span_labels = ["holder", "target", "dse", "attitude", "opinion"]
    mpqa3_span_labels = ["holder", "target:span", "target:entity", "target:event", "target:entity:event", "target", "dse", "attitude", "opinion"]

    for docid in mpqa2_filelist:
        doc = json.load(open(os.path.join(config.DATA_FOLDER, "mpqa2-processed", docid, "tokenized.json")))

        for sentence in doc:
            spans = set()
            for dse in sentence["dse"]:
                for key_prefix in mpqa2_span_labels:
                    key = key_prefix + "-span"
                    span = dse[key]
                    if isinstance(span, list):
                        key_index = mpqa2_span_labels.index(key_prefix)
                        spans.add((key_index, tuple(span)))

            for i, span_x in spans:
                mpqa2_overlap[i, 0] += 1
                overlap_span_labels = set()
                for j, span_y in spans:
                    if j not in overlap_span_labels and (j != i or span_x != span_y) and utils.spans_overlap(span_x, span_y):
                        mpqa2_overlap[i, j + 1] += 1
                        overlap_span_labels.add(j)
    
    for docid in mpqa3_filelist:
        doc = json.load(open(os.path.join(config.DATA_FOLDER, "mpqa3-processed", docid, "tokenized.json")))

        for sentence in doc:
            spans = set()
            for dse in sentence["dse"]:
                for key_prefix in mpqa3_span_labels:
                    span = None
                    if key_prefix.startswith("target") and ":" in key_prefix:
                        target_types = key_prefix.split(":")[1:]
                        if dse["target-type"] in target_types:
                            span = dse["target-span"]
                    else:
                        key = key_prefix + "-span"
                        span = dse[key]
                    if isinstance(span, list):
                        key_index = mpqa3_span_labels.index(key_prefix)
                        spans.add((key_index, tuple(span)))
        
            for i, span_x in spans:
                mpqa3_overlap[i, 0] += 1
                overlap_span_labels = set()
                for j, span_y in spans:
                    if j  not in overlap_span_labels and (j != i or span_x != span_y) and utils.spans_overlap(span_x, span_y):
                        mpqa3_overlap[i, j + 1] += 1
                        overlap_span_labels.add(j)
        
    mpqa2_overlap[:,1:] = 100 * mpqa2_overlap[:,1:]/mpqa2_overlap[:,0].reshape(-1, 1)
    mpqa3_overlap[:,1:] = 100 * mpqa3_overlap[:,1:]/mpqa3_overlap[:,0].reshape(-1, 1)

    mpqa2_overlap_df = pd.DataFrame(mpqa2_overlap, index = mpqa2_span_labels, columns = ["total"] + mpqa2_span_labels)
    mpqa3_overlap_df = pd.DataFrame(mpqa3_overlap, index = mpqa3_span_labels, columns = ["total"] + mpqa3_span_labels)

    print("matrix[label_i, total] is the total number of label_i spans")
    print("matrix[label_i, label_j] is the percentage of label_i spans that overlap with label_j\n")

    print("MPQA 2 overlap stats:")
    print(mpqa2_overlap_df)
    print()

    print("MPQA 3 overlap stats:")
    print(mpqa3_overlap_df)

if __name__=="__main__":
    statistics_overlap()