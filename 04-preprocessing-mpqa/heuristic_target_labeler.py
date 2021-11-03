# author : Sabyasachee Baruah

import os
import json
import spacy
import config
from tqdm import tqdm
import copy
from custom_json_encoder import NoIndent, MyEncoder

def format_json(doc):
    for sentence in doc:
        sentence["tokens"] = NoIndent(sentence["tokens"])
        for dses in [sentence["dse"], sentence["dse-heuristic"]]:
            for dse in dses:
                for prefix in ["dse", "attitude", "holder", "target", "opinion"]:
                    for suffix in ["span", "tokens"]:
                        key = prefix + "-" + suffix
                        if dse[key] is not None:
                            dse[key] = NoIndent(dse[key])

def is_span_contained(smaller_span, larger_span):
    return larger_span[0] <= smaller_span[0] <= smaller_span[1] <= larger_span[1]

def find_targets_heuristic():
    mpqa3_docids = open(os.path.join(config.MPQA3_FOLDER, "doclist")).read().splitlines()
    mpqa2_docids = open(os.path.join(config.MPQA2_FOLDER, "doclist.attitudeSubset")).read().splitlines()

    nlp = spacy.load("en_core_web_sm")

    for i, docids in enumerate([mpqa2_docids, mpqa3_docids]):
        for docid in tqdm(docids, desc="MPQA{}".format(i + 2)):
            doc = json.load(open(os.path.join(config.RESULTS_FOLDER, "mpqa{}-processed".format(i + 2), docid, "tokenized.json")))
            
            for sentence in doc:
                tokens = sentence["tokens"]
                text = sentence["text"]
                
                spacy_sentence = nlp(text)
                spacy_tokens = [token.text for token in spacy_sentence]
                assert tokens == spacy_tokens
                heuristic_dse = []

                for dse in sentence["dse"]:
                    heuristic_target_spans = set()
                    target_span = dse["target-span"]
                    
                    for noun_chunk in spacy_sentence.noun_chunks:
                        noun_chunk_span = (noun_chunk.start, noun_chunk.end)
                        if is_span_contained(noun_chunk_span, target_span):
                            heuristic_target_spans.add((noun_chunk.root.i, noun_chunk.root.i + 1))

                    for token in spacy_sentence:
                        if token.pos_ == "VERB":
                            verb_span = (token.i, token.i + 1)
                            if is_span_contained(verb_span, target_span):
                                heuristic_target_spans.add(verb_span)

                    for heuristic_target_span in heuristic_target_spans:
                        hdse = copy.deepcopy(dse)
                        hdse["target-span"] = list(heuristic_target_span)
                        hdse["target-tokens"] = tokens[slice(*heuristic_target_span)]
                        heuristic_dse.append(hdse)
                
                sentence["dse-heuristic"] = heuristic_dse
        
            format_json(doc)
            json.dump(doc, open(os.path.join(config.RESULTS_FOLDER, "mpqa{}-processed".format(i + 2), docid, "heuristic.json"), "w"), indent=2, sort_keys=True, cls=MyEncoder)

find_targets_heuristic()