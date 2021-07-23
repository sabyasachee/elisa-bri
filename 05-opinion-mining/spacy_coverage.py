import re
import json
import config
import spacy
import os

def find_performance(gold_spans, pred_spans):
    common_spans = gold_spans.intersection(pred_spans)
    precision = len(common_spans)/(1e-23 + len(pred_spans))
    recall = len(common_spans)/(1e-23 + len(gold_spans))
    return "precision = {:5.1f} recall = {:5.1f}".format(100*precision, 100*recall)

def spacy_coverage():
    nlp = spacy.load("en_core_web_sm")
    mpqa3_doc_ids = open(os.path.join(config.MPQA3_FOLDER, "doclist")).read().strip().split("\n")
    mpqa3_docs = [json.load(open(os.path.join(config.MPQA3_PROCESSED_FOLDER, doc_id, "tokenized.json"))) for doc_id in mpqa3_doc_ids]

    target_entities, target_events, inside_target_entities, inside_target_events = set(), set(), set(), set()
    spacy_entities, spacy_nouns, spacy_verbs = set(), set(), set()

    for i, doc in enumerate(mpqa3_docs):
        for j, sentence in enumerate(doc):

            spans = set()
            doc = nlp(re.sub("\s+", " ", sentence["text"]))

            for ex in sentence["dse"]:
                if ex["target-type"] == "span":
                    spans.add(tuple(ex["target"]))
            
            for ex in sentence["dse"]:
                span = tuple(ex["target"])
                if ex["target-type"] == "entity":
                    target_entities.add((i, j, span[0], span[1]))
                elif ex["target-type"] == "event":
                    target_events.add((i, j, span[0], span[1]))
                
                if ex["target-type"] in ["entity", "event"]:
                    for tspan in spans:
                        if tspan[0] <= span[0] <= span[1] <= tspan[1]:
                            if ex["target-type"] == "entity":
                                inside_target_entities.add((i, j, span[0], span[1]))
                            else:
                                inside_target_events.add((i, j, span[0], span[1]))
                            break
            
            for ent in doc.ents:
                span = (ent.start, ent.end - 1)
                for tspan in spans:
                    if tspan[0] <= span[0] <= span[1] <= tspan[1]:
                        spacy_entities.add((i, j, span[0], span[1]))
                        break
            
            for noun_chunk in doc.noun_chunks:
                span = (noun_chunk.root.i, noun_chunk.root.i)
                for tspan in spans:
                    if tspan[0] <= span[0] <= tspan[1]:
                        spacy_nouns.add((i, j, span[0], span[1]))
                        break
            
            for token in doc:
                if token.pos_ == "VERB":
                    span = (token.i, token.i)
                    for tspan in spans:
                        if tspan[0] <= span[0] <= tspan[1]:
                            spacy_verbs.add((i, j, span[0], span[1]))
                            break
            
    print("{:6.2f}% of target entities lie inside some target span".format( 100 * len(inside_target_entities) / len(target_entities) ))
    print("{:6.2f}% of target events   lie inside some target span".format( 100 * len(inside_target_events) / len(target_events) ))
    print("finding target entities with spacy ner        : {}".format(find_performance(target_entities, spacy_entities)))
    print("finding inside target entities with spacy ner : {}".format(find_performance(inside_target_entities, spacy_entities)))
    print("finding target entities with spacy noun       : {}".format(find_performance(target_entities, spacy_nouns)))
    print("finding inside target entities with spacy noun: {}".format(find_performance(inside_target_entities, spacy_nouns)))
    print("finding target events with spacy noun         : {}".format(find_performance(target_events, spacy_verbs)))
    print("finding inside target events with spacy verb  : {}".format(find_performance(inside_target_events, spacy_verbs)))

if __name__ == "__main__":
    spacy_coverage()