import os
import re
import json
import pandas as pd
from tqdm import tqdm
from collections import Counter
from spacy.lang.en import English

import config
from custom_json_encoder import NoIndent, MyEncoder

tokenizer = English().tokenizer

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
            target_type = "span"
            target_tokens = tokens[target_span[0]: target_span[1]]

            if ((isinstance(holder, dict) and holder_index is not None) or holder == "writer" or holder == "implicit"):

                if isinstance(holder, dict):
                    holder_span = [holder_index[1], holder_index[2]]
                    holder_type = "span"
                    holder_tokens = tokens[slice(*holder_span)]
                else:
                    holder_span = None
                    holder_type = holder
                    holder_tokens = []

                attitude_type = attitude["attitude-type"]

                if dse is not None and dse_index is not None and dse_index[0] == target_index[0] and attitude_index is not None and attitude_index[0] == target_index[0]:
                    dse_span = [dse_index[1], dse_index[2]]
                    attitude_span = [attitude_index[1], attitude_index[2]]
                    dse_tokens = tokens[slice(*dse_span)]
                    attitude_tokens = tokens[slice(*attitude_span)]

                elif dse is not None and dse_index is not None and dse_index[0] == target_index[0]:
                    dse_span = [dse_index[1], dse_index[2]]
                    attitude_span = None
                    dse_tokens = tokens[slice(*dse_span)]
                    attitude_tokens = []

                elif attitude_index is not None and attitude_index[0] == target_index[0]:
                    dse_span = None
                    attitude_span = [attitude_index[1], attitude_index[2]]
                    dse_tokens = []
                    attitude_tokens = tokens[slice(*attitude_span)]

                else:
                    dse_span = None
                    attitude_span = None
                    dse_tokens = []
                    attitude_tokens = []

                if dse_span is not None or attitude_span is not None:
                    
                    if (isinstance(holder, dict) and holder_index is not None and holder_index[0] == target_index[0]) or holder == "writer" or holder == "implicit":
                        tokenized_sentences[target_index[0]]["dse"].append({
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

def is_whitespace(text):
    return re.match("^\s*$", text) is not None

def correct_span(span, tokens, n_previous_whitespace_tokens):

    i = span[0]
    j = span[1] - 1
    n = len(tokens)

    while i < n and is_whitespace(tokens[i]):
        i += 1
    
    while 0 <= j and is_whitespace(tokens[j]):
        j -= 1
    
    if 0 <= i <= j < n:
        i = i - n_previous_whitespace_tokens[i]
        j = j - n_previous_whitespace_tokens[j]
        return [i, j + 1]

def remove_whitespace_tokens(tokenized_sentences):

    n_tuples_removed = 0

    for tokenized_sentence in tokenized_sentences:
        
        n_previous_whitespace_tokens = [0 for _ in range(len(tokenized_sentence["tokens"]))]
        non_whitespace_tokens = []
        c = 0
        for i, token in enumerate(tokenized_sentence["tokens"]):
            flag = is_whitespace(token)
            n_previous_whitespace_tokens[i] = c + flag
            if not flag:
                non_whitespace_tokens.append(token)
            c += flag

        new_dse_arr = []

        for dse in tokenized_sentence["dse"]:
            new_dse = dse
            for key in ["dse", "attitude", "holder", "target", "opinion"]:
                span = dse[key + "-span"]
                if span is not None:
                    corrected_span = correct_span(span, tokenized_sentence["tokens"], n_previous_whitespace_tokens)
                    if corrected_span is None:
                        n_tuples_removed += 1
                        break
                    else:
                        new_dse[key + "-span"] = corrected_span
                        new_dse[key + "-tokens"] = non_whitespace_tokens[slice(*corrected_span)]
            else:
                new_dse_arr.append(new_dse)

        tokenized_sentence["dse"] = new_dse_arr
        tokenized_sentence["tokens"] = non_whitespace_tokens
        tokenized_sentence["text"] = re.sub("\s+", " ", tokenized_sentence["text"]).strip()
    
    return n_tuples_removed

def format_json(sentences):
    for sentence in sentences:
        sentence["tokens"] = NoIndent(sentence["tokens"])
        for ex in sentence["dse"]:
            for prefix in ["dse", "attitude", "holder", "target", "opinion"]:
                for suffix in ["span", "tokens"]:
                    key = prefix + "-" + suffix
                    if ex[key] is not None:
                        ex[key] = NoIndent(ex[key])

def tokenize_mpqa2(mpqa2_folder, results_folder):
    doc_ids_file = os.path.join(mpqa2_folder, "doclist.attitudeSubset")
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
        processed_file = os.path.join(results_folder, "mpqa2-processed", doc_id, "processed.json")
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

                    if len(attitude["matched-targets"]) > 0:
                        for target in attitude["matched-targets"]:
                            target_index = find_token_index_of_span(target["sentence-index"], target["span"], tokenized_sentences)
                            doc_records.append([dse_index, holder_type, holder_index, attitude_index, attitude_is_sentiment, "target", target_index])
                            doc_tuples.append([dse, holder, attitude, target, dse_index, holder_index, attitude_index, target_index])
                        
                    else:
                        doc_records.append([dse_index, holder_type, holder_index, attitude_index, attitude_is_sentiment, None, None])
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
        
        doc_n_tuples_removed = remove_whitespace_tokens(tokenized_sentences)
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
        
        format_json(new_tokenized_sentences)

        folder = os.path.join(results_folder, "mpqa2-processed", doc_id)
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, "tokenized.json")
        json.dump(new_tokenized_sentences, open(file, "w"), indent=2, sort_keys=True, cls=MyEncoder)
    
    print("{} tuples".format(n_tuples))
    print("{} sentiment tuples".format(n_sentiment_tuples))
    print("{} tuples have holder within one sentence of the target".format(n_two_sentence_tuples))
    print("{} sentiment tuples have holder within one sentence of the target".format(n_sentiment_two_sentence_tuples))
    print("{} tuples were tokenized".format(n_tokenized_tuples))
    print("{} sentiment tuples were tokenized".format(n_sentiment_tokenized_tuples))
    print("{} tokenized tuples remain after removing whitespace tokens".format(n_tokenized_tuples_after_correction))

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
    
    folder = os.path.join(results_folder, "mpqa2-processed")
    os.makedirs(folder, exist_ok=True)
    file = os.path.join(folder, "stats.csv")
    df.to_csv(file, index=True)

if __name__ == "__main__":
    tokenize_mpqa2(config.MPQA2_FOLDER, config.RESULTS_FOLDER)