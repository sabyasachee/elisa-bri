import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from spacy.lang.en import English
import re

import config
from custom_json_encoder import NoIndent, MyEncoder

tokenizer = English().tokenizer

def is_span_overlap(span_x, span_y):
    return span_x[0] <= span_y[1] and span_y[0] <= span_x[1]

def remove_overlapping_spans_in_order(spans):
    new_spans = []
    prev_span = None
    for span in spans:
        if prev_span is not None and is_span_overlap(prev_span, span):
            continue
        new_spans.append(span)
        prev_span = span
    return new_spans

def create_label_arr(spans, length):
    label_arr = np.full(length, "O")
    for span in spans:
        label_arr[span[0]] = "B"
        label_arr[span[0] + 1: span[1] + 1] = "I"
    return label_arr.tolist()

def create_label_arrs(sentences):
    for sentence in sentences:
        opinion_span_dict = defaultdict(lambda: np.zeros(3))
        holder_span_dict = defaultdict(lambda: np.zeros(3))
        entity_span_dict = defaultdict(lambda: np.zeros(3))
        event_span_dict = defaultdict(lambda: np.zeros(3))

        for ex in sentence["dse"]:
            if ex["target-type"] == "entity" or ex["target-type"] == "event":
                
                score = [1, ex["attitude-type"] == "sentiment-pos", ex["attitude-type"] == "sentiment-neg"]
                
                opinion_span = ex["opinion"]
                opinion_span_dict[tuple(opinion_span)] += score

                if ex["holder-type"] == "span":
                    holder_span = ex["holder"]
                    holder_span_dict[tuple(holder_span)] += score

                if ex["target-type"] == "entity":
                    entity_span_dict[tuple(ex["target"])] += score
                else:
                    event_span_dict[tuple(ex["target"])] += score
        
        opinion_spans = sorted(opinion_span_dict.keys(), key = lambda span: tuple(opinion_span_dict[span]), reverse=True)
        holder_spans = sorted(holder_span_dict.keys(), key = lambda span: tuple(holder_span_dict[span]), reverse=True)
        entity_spans = sorted(entity_span_dict.keys(), key = lambda span: tuple(entity_span_dict[span]), reverse=True)
        event_spans = sorted(event_span_dict.keys(), key = lambda span: tuple(event_span_dict[span]), reverse=True)

        opinion_spans = remove_overlapping_spans_in_order(opinion_spans)
        holder_spans = remove_overlapping_spans_in_order(holder_spans)
        entity_spans = remove_overlapping_spans_in_order(entity_spans)
        event_spans = remove_overlapping_spans_in_order(event_spans)

        opinion_spans = sorted(opinion_spans)
        holder_spans = sorted(holder_spans)
        entity_spans = sorted(entity_spans)
        event_spans = sorted(event_spans)

        opinion_label_arr = create_label_arr(opinion_spans, len(sentence["tokens"]))
        holder_label_arr = create_label_arr(holder_spans, len(sentence["tokens"]))
        entity_label_arr = create_label_arr(entity_spans, len(sentence["tokens"]))
        event_label_arr = create_label_arr(event_spans, len(sentence["tokens"]))

        sentence["dse-opinion"] = opinion_label_arr
        sentence["dse-holder"] = holder_label_arr
        sentence["dse-entity"] = entity_label_arr
        sentence["dse-event"] = event_label_arr

def find_token_index_of_span(sentence_index, span, tokenized_sentences):
    if sentence_index is not None and span is not None and len(tokenized_sentences) > 0:
        tokenized_sentence = tokenized_sentences[sentence_index]
        token_offsets = tokenized_sentence["token-offsets"]
        sentence_offset = tokenized_sentence["offset"]

        i = [k for k, (x, _) in enumerate(token_offsets) if span["start"] == x + sentence_offset]
        j = [k for k, (_, y) in enumerate(token_offsets) if span["end"] == y + sentence_offset]
        
        if len(i) == 1 and len(j) == 1:
            return [sentence_index, i[0], j[0] + 1]

def is_equal_token_indexes(token_index_1, token_index_2):
    return token_index_1 is not None and token_index_2 is not None and token_index_1[0] == token_index_2[0]

def create_tokenized_tuples(doc_tuples, tokenized_sentences):

    n_tuples = len(doc_tuples)
    n_sentiment_tuples = sum([attitude is not None and attitude["attitude-type"] is not None and attitude["attitude-type"].startswith("sentiment") for _, _, attitude, _, _, _, _, _ in doc_tuples])
    n_two_sentence_tuples = 0
    n_sentiment_two_sentence_tuples = 0
    n_tokenized_tuples = 0
    n_sentiment_tokenized_tuples = 0

    for dse, holder, attitude, target, dse_index, holder_index, attitude_index, target_index in doc_tuples:

        if attitude is not None and attitude["attitude-type"] is not None and target is not None and target_index is not None and holder is not None:

            tokenized_sentence = tokenized_sentences[target_index[0]]
            tokens = tokenized_sentence["tokens"]

            target_span = [target_index[1], target_index[2]]
            target_tokens = tokens[target_span[0]: target_span[1]]
            if "matched-etargets" in target:
                target_type = "span"
            else:
                target_type = target["type"]

            if target_type in ["span", "entity", "event"] and ((isinstance(holder, dict) and holder_index is not None) or holder == "writer" or holder == "implicit"):

                if isinstance(holder, dict):
                    holder_span = [holder_index[1], holder_index[2]]
                    holder_type = "span"
                    holder_tokens = tokens[holder_span[0]: holder_span[1]]
                else:
                    holder_span = None
                    holder_type = holder
                    holder_tokens = []

                attitude_type = attitude["attitude-type"]

                if dse is not None and dse_index is not None and dse_index[0] == target_index[0] and attitude_index is not None and attitude_index[0] == target_index[0]:
                    dse_span = [dse_index[1], dse_index[2]]
                    attitude_span = [attitude_index[1], attitude_index[2]]
                    dse_tokens = tokens[dse_span[0]: dse_span[1]]
                    attitude_tokens = tokens[attitude_span[0]: attitude_span[1]]
                
                elif dse is not None and dse_index is not None and dse_index[0] == target_index[0]:
                    dse_span = [dse_index[1], dse_index[2]]
                    attitude_span = None
                    dse_tokens = tokens[dse_span[0]: dse_span[1]]
                    attitude_tokens = []

                elif attitude_index is not None and attitude_index[0] == target_index[0]:
                    dse_span = None
                    attitude_span = [attitude_index[1], attitude_index[2]]
                    dse_tokens = []
                    attitude_tokens = tokens[attitude_span[0]: attitude_span[1]]

                else:
                    dse_span = None
                    attitude_span = None
                    dse_tokens = []
                    attitude_tokens = []

                if dse_span is not None or attitude_span is not None:
                    if (isinstance(holder, dict) and holder_index is not None and holder_index[0] == target_index[0]) or holder == "writer" or holder == "implicit":
                        tokenized_sentence["dse"].append({
                            "dse-span": dse_span,
                            "dse-tokens": dse_tokens,
                            "attitude-span": attitude_span,
                            "attitude-tokens": attitude_tokens,
                            "opinion-span": dse_span if dse_span is not None else attitude_span,
                            "opinion-tokens": dse_tokens if dse_span is not None else attitude_tokens,
                            "holder-type": holder_type,
                            "holder-span": holder_span,
                            "holder-tokens": holder_tokens,
                            "target-type": target_type,
                            "target-span": target_span,
                            "target-tokens": target_tokens,
                            "attitude-type": attitude_type
                        })
                        n_tokenized_tuples += 1
                        n_sentiment_tokenized_tuples += attitude_type.startswith("sentiment")
                    if (isinstance(holder, dict) and holder_index is not None and abs(holder_index[0] - target_index[0]) < 2) or holder == "writer" or holder == "implicit":
                        n_two_sentence_tuples += 1
                        n_sentiment_two_sentence_tuples += attitude_type.startswith("sentiment")
        
    return n_tuples, n_sentiment_tuples, n_two_sentence_tuples, n_sentiment_two_sentence_tuples, n_tokenized_tuples, n_sentiment_tokenized_tuples

def correct_span(span, tokens, number_of_newline_characters):

    while span[0] < span[1] and re.match("^\s+$", tokens[span[0]]):
        span[0] += 1
    
    while span[0] < span[1] and re.match("^\s+$", tokens[span[1]-1]):
        span[1] -= 1
    
    if span[0] < span[1]:
        span[0] -= number_of_newline_characters[span[0]]
        span[1] -= number_of_newline_characters[span[1]-1]
        return span

def remove_newline_characters(tokenized_sentences):

    n_tuples_removed = 0

    for tokenized_sentence in tokenized_sentences:
        
        number_of_newline_characters = [0 for _ in range(len(tokenized_sentence["tokens"]))]
        c = 0
        for i, token in enumerate(tokenized_sentence["tokens"]):
            number_of_newline_characters[i] = c + (re.match("^\s+$", token) is not None)
            c += re.match("^\s+$", token) is not None

        new_dse_arr = []

        for dse in tokenized_sentence["dse"]:
            new_dse = dse
            for key in ["dse-span", "attitude-span", "holder-span", "target-span", "opinion-span"]:
                span = dse[key]
                if span is not None:
                    corrected_span = correct_span(span, tokenized_sentence["tokens"], number_of_newline_characters)
                    if corrected_span is None:
                        n_tuples_removed += 1
                        break
                    else:
                        new_dse[key] = corrected_span
            else:
                new_dse_arr.append(new_dse)

        tokenized_sentence["dse"] = new_dse_arr
        tokenized_sentence["tokens"] = [token for token in tokenized_sentence["tokens"] if not re.match("^\s+$", token)]
    
    return n_tuples_removed

def format_json(sentences):
    for sentence in sentences:
        sentence["tokens"] = NoIndent(sentence["tokens"])
        # sentence["dse-opinion"] = NoIndent(sentence["dse-opinion"])
        # sentence["dse-entity"] = NoIndent(sentence["dse-entity"])
        # sentence["dse-event"] = NoIndent(sentence["dse-event"])
        # sentence["dse-holder"] = NoIndent(sentence["dse-holder"])
        for ex in sentence["dse"]:
            for prefix in ["dse", "attitude", "holder", "target", "opinion"]:
                for suffix in ["span", "tokens"]:
                    key = prefix + "-" + suffix
                    if ex[key] is not None:
                        ex[key] = NoIndent(ex[key])

def tokenize_mpqa3(mpqa3_folder, results_folder):
    doc_ids_file = os.path.join(mpqa3_folder, "doclist")
    doc_ids = open(doc_ids_file).read().strip().split("\n")
    
    records = []
    n_tuples = 0
    n_sentiment_tuples = 0
    n_two_sentence_tuples = 0
    n_sentiment_two_sentence_tuples = 0
    n_tokenized_tuples = 0
    n_sentiment_tokenized_tuples = 0
    n_tokenized_tuples_after_correction = 0

    for doc_id in tqdm(doc_ids):
        processed_file = os.path.join(results_folder, "mpqa3-processed", doc_id, "processed.json")
        processed = json.load(open(processed_file))

        tokenized_sentences = []
        doc_records = []
        doc_tuples = []

        for sentence in processed["sentences"]:
            tokenized_sentence = {"text": sentence["text"], "offset": sentence["span"]["start"]}
            tokens = tokenizer(sentence["text"])
            tokenized_sentence["tokens"] = [token.text for token in tokens]
            tokenized_sentence["token-offsets"] = [[token.idx, token.idx + len(token)] for token in tokens]
            tokenized_sentence["dse"] = []
            tokenized_sentences.append(tokenized_sentence)

        for dse in processed["dse"]:
            dse_index = find_token_index_of_span(dse["sentence-index"], dse["span"], tokenized_sentences)

            holder = dse["matched-source"]
            holder_index = None
            
            if isinstance(holder, dict):
                holder_index = find_token_index_of_span(holder["sentence-index"], holder["span"], tokenized_sentences)
                holder_type = "agent"
            else:
                holder_type = holder

            if len(dse["matched-attitudes"]) > 0:
                for attitude in dse["matched-attitudes"]:
                    attitude_index = find_token_index_of_span(attitude["sentence-index"], attitude["span"], tokenized_sentences)
                    attitude_is_sentiment = attitude["attitude-type"] is not None and attitude["attitude-type"].startswith("sentiment")

                    if attitude["matched-targetframe"] is not None and len(attitude["matched-targetframe"]["matched-stargets"]) + len(attitude["matched-targetframe"]["matched-etargets"]) > 0:
                        stargets = attitude["matched-targetframe"]["matched-stargets"]
                        etargets = attitude["matched-targetframe"]["matched-etargets"] + [etarget for starget in stargets for etarget in starget["matched-etargets"]]

                        for starget in stargets:
                            starget_index = find_token_index_of_span(starget["sentence-index"], starget["span"], tokenized_sentences)
                            doc_records.append([dse_index, holder_type, holder_index, attitude_index, attitude_is_sentiment, "starget", starget_index])
                            doc_tuples.append([dse, holder, attitude, starget, dse_index, holder_index, attitude_index, starget_index])
                        
                        for etarget in etargets:
                            etarget_index = find_token_index_of_span(etarget["sentence-index"], etarget["span"], tokenized_sentences)
                            doc_records.append([dse_index, holder_type, holder_index, attitude_index, attitude_is_sentiment, etarget["type"], etarget_index])
                            doc_tuples.append([dse, holder, attitude, etarget, dse_index, holder_index, attitude_index, etarget_index])

                    else:
                        doc_records.append([dse_index, holder_type, holder_index, attitude_index, False, None, None])
                        doc_tuples.append([dse, holder, attitude, None, dse_index, holder_index, attitude_index, None])

            else:
                doc_records.append([dse_index, holder_type, holder_index, None, False, None, None])
                doc_tuples.append([dse, holder, None, None, dse_index, holder_index, None, None])

        doc_n_tuples, doc_n_sentiment_tuples, doc_n_two_sentence_tuples, doc_n_sentiment_two_sentence_tuples, doc_n_tokenized_tuples, doc_n_sentiment_tokenized_tuples = create_tokenized_tuples(doc_tuples, tokenized_sentences)
        n_tuples += doc_n_tuples
        n_sentiment_tuples += doc_n_sentiment_tuples
        n_two_sentence_tuples += doc_n_two_sentence_tuples
        n_sentiment_two_sentence_tuples += doc_n_sentiment_two_sentence_tuples
        n_tokenized_tuples += doc_n_tokenized_tuples
        n_sentiment_tokenized_tuples += doc_n_sentiment_tokenized_tuples
        
        doc_n_tuples_removed = remove_newline_characters(tokenized_sentences)
        n_tokenized_tuples_after_correction += doc_n_tokenized_tuples - doc_n_tuples_removed

        records.extend(doc_records)

        new_tokenized_sentences = []

        for tokenized_sentence in tokenized_sentences:
            new_tokenized_sentence = {
                "text": tokenized_sentence["text"],
                "tokens": tokenized_sentence["tokens"],
                "dse": tokenized_sentence["dse"]
            }
            new_tokenized_sentences.append(new_tokenized_sentence)

        # create_label_arrs(new_tokenized_sentences)

        format_json(new_tokenized_sentences)

        folder = os.path.join(results_folder, "mpqa3-processed", doc_id)
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, "tokenized.json")
        json.dump(new_tokenized_sentences, open(file, "w"), indent=2, sort_keys=True, cls=MyEncoder)
    
    print("{} tuples".format(n_tuples))
    print("{} sentiment tuples".format(n_sentiment_tuples))
    print("{} tuples have holder within one sentence of the target".format(n_two_sentence_tuples))
    print("{} sentiment tuples have holder within one sentence of the target".format(n_sentiment_two_sentence_tuples))
    print("{} tuples were tokenized".format(n_tokenized_tuples))
    print("{} sentiment tuples were tokenized".format(n_sentiment_tokenized_tuples))
    print("{} tokenized tuples remain after removing newline tokens".format(n_tokenized_tuples_after_correction))

    new_records = []

    for dse_index, holder_type, holder_index, attitude_index, attitude_is_sentiment, target_type, target_index in records:
        dse_holder = is_equal_token_indexes(dse_index, holder_index)
        dse_attitude = is_equal_token_indexes(dse_index, attitude_index)
        dse_target = is_equal_token_indexes(dse_index, target_index)
        holder_attitude = is_equal_token_indexes(holder_index, attitude_index)
        holder_target = is_equal_token_indexes(holder_index, target_index)
        attitude_target = is_equal_token_indexes(attitude_index, target_index)

        new_records.append((dse_index is not None, holder_index is not None, attitude_index is not None, target_index is not None, dse_holder, dse_attitude, dse_target, holder_attitude, holder_target, attitude_target, attitude_is_sentiment, holder_type, target_type))
    
    records_distribution = Counter(new_records)
    data = []
    for key, count in records_distribution.items():
        data.append(list(key) + [count])
    columns = ["DSE", "HOL", "ATT", "TGT", "DSE=HOL", "DSE=ATT", "DSE=TGT", "HOL=ATT", "HOL=TGT", "ATT=TGT", "SENTIMENT", "HOL-TYP", "TGT-TYP", "count"]
    df = pd.DataFrame(data, columns=columns)
    df = df.set_index(columns[:-1])
    df = df.sort_values("count", ascending=False)
    
    folder = os.path.join(results_folder, "mpqa3-processed")
    os.makedirs(folder, exist_ok=True)
    file = os.path.join(folder, "stats.csv")
    df.to_csv(file, index=True)

if __name__ == "__main__":
    tokenize_mpqa3(config.MPQA3_FOLDER, config.RESULTS_FOLDER)