# author : Sabyasachee Baruah

import os
import json
import config
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def create_histogram(arr, var, savefile, mpqa3 = False):
    '''
    Create histogram from `arr`. `var` is the variable. The histogram picture is saved to `savefile`.
    Set `mpqa3 = TRUE` if `var` is from MPQA 3 docs
    '''

    mean = np.mean(arr)
    std = np.std(arr)
    max = np.max(arr)
    min = np.min(arr)
    median = np.median(arr)
    ninety_percentile = np.quantile(arr, 0.9)

    plt.figure(figsize=(10,10))
    plt.rcParams.update({"font.size": 16})

    n_unique = len(set(arr))
    bins = np.min([n_unique, 10])

    plt.hist(arr, bins=bins)
    ax = plt.gca()

    plt.text(0.7, 0.8, "$\mu = {:.2f}, \sigma = {:.2f}$".format(mean, std), transform=ax.transAxes)
    plt.text(0.7, 0.7, "Median = {:.2f}, 90%tile = {:.2f}".format(median, ninety_percentile), transform=ax.transAxes)
    plt.text(0.7, 0.6, "range = [{}, {}]".format(min, max), transform=ax.transAxes)

    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.title("Histogram of {} in MPQA{}".format(var, 2 + mpqa3))

    plt.savefig(savefile)
    plt.close()

def find_descriptive_statistics():
    mpqa2_docids_file = os.path.join(config.MPQA2_DATA_FOLDER, "doclist.attitudeSubset")
    mpqa2_docids = open(mpqa2_docids_file).read().splitlines()
    mpqa2_docs = [json.load(open(os.path.join(config.MPQA2_PROCESSED_FOLDER, docid, "tokenized.json"))) for docid in mpqa2_docids]

    mpqa3_docids_file = os.path.join(config.MPQA3_DATA_FOLDER, "doclist")
    mpqa3_docids = open(mpqa3_docids_file).read().splitlines()
    mpqa3_docs = [json.load(open(os.path.join(config.MPQA3_PROCESSED_FOLDER, docid, "tokenized.json"))) for docid in mpqa3_docids]

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)

    sentence_lengths = [[], []]
    holder_lengths = [[], []]
    target_span_lengths = [[], []]
    target_entity_lengths = []
    target_event_lengths = []
    expression_lengths = [[], []]
    dse_lengths = [[], []]

    n_opinion_tuples_per_sentence = [[], []]
    n_holders_per_sentence = [[], []]
    n_target_spans_per_sentence = [[], []]
    n_target_entities_per_sentence = []
    n_target_events_per_sentence = []

    attitude_distribution = [defaultdict(int), defaultdict(int)]
    holder_type_distribution = [{"explicit": 0, "implicit": 0}, {"explicit": 0, "implicit": 0}]

    for d, docs in enumerate([mpqa2_docs, mpqa3_docs]):
        for doc in tqdm(docs, desc="MPQA{}".format(d + 2)):
            
            for sentence in doc:
                tokens = sentence["tokens"]
                n_previous_tokens = np.zeros(len(tokens) + 1)
                wordpiece_tokens = []
                
                for i, token in enumerate(tokens):
                    wordpiece_tokens_from_single_token = tokenizer.tokenize(token)
                    wordpiece_tokens.extend(wordpiece_tokens_from_single_token)
                    n_previous_tokens[i + 1] = n_previous_tokens[i] + len(wordpiece_tokens_from_single_token)
                
                # assert wordpiece_tokens == tokenizer.tokenize(sentence["text"])

                for dse in sentence["dse"]:
                    for key in ["opinion", "holder", "target", "dse"]:
                        span_key = key + "-span"
                        if dse[span_key]:
                            start, end = dse[span_key]
                            new_start = n_previous_tokens[start]
                            new_end = n_previous_tokens[end]
                            dse[span_key] = [new_start, new_end]
                
                holders = set()
                target_spans = set()
                target_entities = set()
                target_events = set()
                expressions = set()
                dse_spans = set()

                n_opinions = 0

                for dse in sentence["dse"]:
                    if dse["holder-type"] == "span":
                        holders.add(tuple(dse["holder-span"]))

                    if dse["target-type"] == "span":
                        target_spans.add(tuple(dse["target-span"]))
                    elif dse["target-type"] == "entity":
                        target_entities.add(tuple(dse["target-span"]))
                    else:
                        target_events.add(tuple(dse["target-span"]))
                    
                    expressions.add(tuple(dse["opinion-span"]))
                    
                    if dse["dse-span"]:
                        dse_spans.add(tuple(dse["dse-span"]))
                
                    if dse["target-type"] == "span":
                        if dse["holder-type"] == "span":
                            holder_type_distribution[d]["explicit"] += 1
                        else:
                            holder_type_distribution[d]["implicit"] += 1
                        
                        attitude_distribution[d][dse["attitude-type"]] += 1
                        n_opinions += 1
                
                sentence_lengths[d].append(len(wordpiece_tokens))

                holder_lengths[d].extend([span[1] - span[0] for span in holders])
                target_span_lengths[d].extend([span[1] - span[0] for span in target_spans])
                expression_lengths[d].extend([span[1] - span[0] for span in expressions])
                dse_lengths[d].extend([span[1] - span[0] for span in dse_spans])

                n_opinion_tuples_per_sentence[d].append(n_opinions)
                n_holders_per_sentence[d].append(len(holders))
                n_target_spans_per_sentence[d].append(len(target_spans))
                
                if d == 1:
                    target_entity_lengths.extend([span[1] - span[0] for span in target_entities])
                    target_event_lengths.extend([span[1] - span[0] for span in target_events])
                    n_target_entities_per_sentence.append(len(target_entities))
                    n_target_events_per_sentence.append(len(target_events))

    for d in [0, 1]:
        create_histogram(sentence_lengths[d], "sentence length", os.path.join(config.RESULTS_FOLDER, "mpqa{}-sentence-length.png".format(d + 2)), mpqa3 = bool(d))
        create_histogram(holder_lengths[d], "holder length", os.path.join(config.RESULTS_FOLDER, "mpqa{}-holder-length.png".format(d + 2)), mpqa3 = bool(d))
        create_histogram(target_span_lengths[d], "target length", os.path.join(config.RESULTS_FOLDER, "mpqa{}-span-target-length.png".format(d + 2)), mpqa3 = bool(d))
        create_histogram(expression_lengths[d], "opinion expression length", os.path.join(config.RESULTS_FOLDER, "mpqa{}-opinion-expression-length.png".format(d + 2)), mpqa3 = bool(d))
        create_histogram(dse_lengths[d], "direct subjective expression length", os.path.join(config.RESULTS_FOLDER, "mpqa{}-direct-subjective-expression-length.png".format(d + 2)), mpqa3 = bool(d))
        create_histogram(n_opinion_tuples_per_sentence[d], "number of opinion tuples per sentence", os.path.join(config.RESULTS_FOLDER, "mpqa{}-n-opinion-tuples-per-sentence.png".format(d + 2)), mpqa3 = bool(d))
        create_histogram(n_holders_per_sentence[d], "number of holders per sentence", os.path.join(config.RESULTS_FOLDER, "mpqa{}-n-holders-per-sentence.png".format(d + 2)), mpqa3 = bool(d))
        create_histogram(n_target_spans_per_sentence[d], "number of span targets per sentence", os.path.join(config.RESULTS_FOLDER, "mpqa{}-n-span-targets-per-sentence.png".format(d + 2)), mpqa3 = bool(d))
        print("MPQA{} attitude distribution = {}".format(d + 2, attitude_distribution[d]))
        print("MPQA{} holder-type distribution = {}".format(d + 2, holder_type_distribution[d]))
    
    create_histogram(target_entity_lengths, "entity target length", os.path.join(config.RESULTS_FOLDER, "mpqa3-entity-target-length.png"), mpqa3 = True)
    create_histogram(target_event_lengths, "event target length", os.path.join(config.RESULTS_FOLDER, "mpqa3-event-target-length.png"), mpqa3 = True)
    create_histogram(n_target_entities_per_sentence, "number of entity targets per sentence", os.path.join(config.RESULTS_FOLDER, "mpqa3-n-entity-targets-per-sentence.png"), mpqa3 = True)
    create_histogram(n_target_events_per_sentence, "number of event targets per sentence", os.path.join(config.RESULTS_FOLDER, "mpqa3-n-event-targets-per-sentence.png"), mpqa3 = True)

find_descriptive_statistics()     