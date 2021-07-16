import os
import config
import json
import numpy as np

def is_span_overlap(span_x, span_y):
    return span_x is not None and span_y is not None and span_x[0] <= span_y[1] and span_y[0] <= span_x[1]

def num_span_overlaps(spans_x, spans_y = None):
    n, N = 0, 0
    spans_x = list(set([tuple(span) for span in spans_x if span is not None]))
    spans_x.sort()

    if spans_y is not None:
        spans_y = list(set([tuple(span) for span in spans_y if span is not None]))
        for span_x in spans_x:
            for span_y in spans_y:
                n += is_span_overlap(span_x, span_y)
                N += 1

    else:
        for i in range(len(spans_x)):
            for j in range(i + 1, len(spans_x)):
                n += is_span_overlap(spans_x[i], spans_x[j])
                N += 1

    return [n, N]

def update_overlap_matrix(overlap, spans):
    for i in range(len(spans)):
        for j in range(i, len(spans)):
            if i == j:
                overlap[i, j] += num_span_overlaps(spans[i])
            else:
                overlap[i, j] += num_span_overlaps(spans[i], spans[j])

def print_overlap_stats(mpqa2_folder, mpqa3_folder, results_folder):
    mpqa2_doc_ids_file = os.path.join(mpqa2_folder, "doclist.attitudeSubset")
    mpqa3_doc_ids_file = os.path.join(mpqa3_folder, "doclist")
    mpqa2_doc_ids = open(mpqa2_doc_ids_file).read().strip().split("\n")
    mpqa3_doc_ids = open(mpqa3_doc_ids_file).read().strip().split("\n")

    mpqa2_types = ["dse", "attitude", "holder", "starget"]
    mpqa3_types = ["dse", "attitude", "holder", "entity", "event", "starget"]
    mpqa2_overlap = np.zeros((len(mpqa2_types), len(mpqa2_types), 2), dtype = np.int)
    mpqa2_overlap_when_attitude_is_sentiment = np.zeros((len(mpqa2_types), len(mpqa2_types), 2), dtype = np.int)
    mpqa3_overlap = np.zeros((len(mpqa3_types), len(mpqa3_types), 2), dtype = np.int)
    mpqa3_overlap_when_attitude_is_sentiment = np.zeros((len(mpqa3_types), len(mpqa3_types), 2), dtype = np.int)

    for doc_id in mpqa2_doc_ids:
        doc_file = os.path.join(results_folder, "mpqa2-processed", doc_id, "tokenized.json")
        doc = json.load(open(doc_file))

        for sentence in doc:
            list_of_spans = [[], [], [], []]
            list_of_sentiment_spans = [[], [], [], []]

            for ex in sentence["dse"]:
                list_of_spans[0].append(ex["dse"])
                list_of_spans[1].append(ex["attitude"])
                list_of_spans[2].append(ex["holder"])
                list_of_spans[3].append(ex["target"])

                if ex["attitude-type"].startswith("sentiment"):
                    list_of_sentiment_spans[0].append(ex["dse"])
                    list_of_sentiment_spans[1].append(ex["attitude"])
                    list_of_sentiment_spans[2].append(ex["holder"])
                    list_of_sentiment_spans[3].append(ex["target"])

            update_overlap_matrix(mpqa2_overlap, list_of_spans)
            update_overlap_matrix(mpqa2_overlap_when_attitude_is_sentiment, list_of_sentiment_spans)
    
    for doc_id in mpqa3_doc_ids:
        doc_file = os.path.join(results_folder, "mpqa3-processed", doc_id, "tokenized.json")
        doc = json.load(open(doc_file))

        for sentence in doc:
            list_of_spans = [[], [], [], [], [], []]
            list_of_sentiment_spans = [[], [], [], [], [], []]

            for ex in sentence["dse"]:
                list_of_spans[0].append(ex["dse"])
                list_of_spans[1].append(ex["attitude"])
                list_of_spans[2].append(ex["holder"])
                
                if ex["target-type"] == "entity":
                    list_of_spans[3].append(ex["target"])
                elif ex["target-type"] == "event":
                    list_of_spans[4].append(ex["target"])
                else:
                    list_of_spans[5].append(ex["target"])
            
                if ex["attitude-type"].startswith("sentiment"):
                    list_of_sentiment_spans[0].append(ex["dse"])
                    list_of_sentiment_spans[1].append(ex["attitude"])
                    list_of_sentiment_spans[2].append(ex["holder"])
                    
                    if ex["target-type"] == "entity":
                        list_of_sentiment_spans[3].append(ex["target"])
                    elif ex["target-type"] == "event":
                        list_of_sentiment_spans[4].append(ex["target"])
                    else:
                        list_of_sentiment_spans[5].append(ex["target"])

            update_overlap_matrix(mpqa3_overlap, list_of_spans)
            update_overlap_matrix(mpqa3_overlap_when_attitude_is_sentiment, list_of_sentiment_spans)

    print("mpqa2: number of overlaps within a sentence")
    for i in range(len(mpqa2_types)):
        for j in range(i, len(mpqa2_types)):
            print("\t{:10s} x {:10s} = {:4d}/{:4d} ({:.2f}%)".format(mpqa2_types[i], mpqa2_types[j], mpqa2_overlap[i, j, 0], mpqa2_overlap[i, j, 1], 100 * mpqa2_overlap[i, j, 0] / mpqa2_overlap[i, j, 1]))
    
    print("\n")

    print("mpqa3: number of overlaps within a sentence")
    for i in range(len(mpqa3_types)):
        for j in range(i, len(mpqa3_types)):
            print("\t{:10s} x {:10s} = {:4d}/{:4d} ({:.2f}%)".format(mpqa3_types[i], mpqa3_types[j], mpqa3_overlap[i, j, 0], mpqa3_overlap[i, j, 1], 100 * mpqa3_overlap[i, j, 0] / mpqa3_overlap[i, j, 1]))

if __name__ == "__main__":
    print_overlap_stats(config.MPQA2_FOLDER, config.MPQA3_FOLDER, config.RESULTS_FOLDER)