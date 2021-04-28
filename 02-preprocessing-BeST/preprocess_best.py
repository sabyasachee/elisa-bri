'''
author = Sabyasachee Baruah

This script preprocesses the TAC KBP BeST data (2016-2017) and creates json files.
We only process sentiment annotations of newswire data whose target is either an event or an entity.
A single json file is created for each language: english, chinese, spanish.
The directory locations of the BeST data and the results directory are specified in config.py.
You can set the global variable best_dir to preprocess either 2016 or 2017 data.

The json file for a particular language has the following format:
{
    "processed": [{
        "doc_id":str,
        "tokenized_texts": [{
            ?"tokens": [str],
            ?"offsets": [int],
            "type": "headline"|"author"|"body",
            "text": str,
            "source_offset": int| -1
        }]
        "annotations": [{
            "target": {
                "type": "entity"|"event",
                "ere_id": str,
                "offset": int,
                "length": int,
                "index": None|(int,int,-1)|(int,int,int!=-1)
            },
            "source": {
                "type": "mention"|"author"|"other",
                ?"ere_id": str,
                ?"offset": int,
                ?"length": int,
                ?"index": None|(int,int,-1)|(int,int,int!=-1)
                ?"valid_nearest_index": None|(int,int,-1)|(int,int,int!=-1)
            },
            "polarity": "pos"|"neg"
        }]
    }],
    "stats": {str:int}
}

tokenized_texts is a list of sentences.
Each sentence contains tokens and their offsets in the source xml file.
The source_offset is the offset of the full text in the source xml file.
source_offset always equals offsets[0], unless we were not able to find the text in the source xml file.
source_offset then is -1, and the tokens and offsets keys are then absent (indicated by ?).

annotations is a list of sentiment annotations.
Each sentiment annotation contains a target, source and polarity.
BeST data does not contain opinion expression annotations.

The target can be either entity or event, indicated by its type.
ere_id is an identifier for the target in the rich_ere.xml file.
index is the token span index of the target in the tokenized texts.
index = (i, j, k) means that the target corresponds to tokens j to k (inclusive) in sentence i of tokenized_texts.
index is None if we could not find any token whose offset equaled the target offset.
index = (i, j, -1) means that the target starts from token j in sentence i, but crosses sentence boundaries.

The source can be either a mention (entity), author or other.
If source is an entity mention, we define ere_id, offset, length, index similarly as target.
valid_nearest_index is the closest index to the target index of a coreferring mention of the source entity mention.
valid_nearest_index is also only defined if the source type is mention.

polarity is the sentiment orientation. It is +ve or -ve.

stats contain a host of data statistics about the preprocessing process.
'''

import os
import re
import xml.etree.ElementTree as ET
import config
import spacy
import html
from tqdm import tqdm
from collections import defaultdict
import json

best_dir = config.BEST_2016_DIR

langs = ["eng", "cmn", "spa"]
spacy_models = [spacy.load("en_core_web_sm"), spacy.load("zh_core_web_sm"), spacy.load("es_core_news_sm")]
results_dir = os.path.join(config.RESULTS_DIR, os.path.basename(best_dir.rstrip("/")))
os.makedirs(results_dir, exist_ok=True)

def find_index(offset, length, tokenized_texts):
    for i, data in enumerate(tokenized_texts):
        if data["source_offset"] is not None:
            for j in range(len(data["tokens"])):
                if offset == data["offsets"][j]:
                    k = j
                    while k < len(data["tokens"]):
                        span_length = data["offsets"][k] - data["offsets"][j] + len(data["tokens"][k])
                        if span_length == length:
                            break
                        k += 1
                    if k < len(data["tokens"]):
                        return [i, j, k]
                    return [i, j, -1]

for lang, spacy_model in zip(langs, spacy_models):
    source_dir = os.path.join(best_dir, "data", lang, "source")
    ere_dir = os.path.join(best_dir, "data", lang, "ere")
    annotation_dir = os.path.join(best_dir, "data", lang, "annotation")

    doc_ids = [f.rstrip(".xml") for f in os.listdir(source_dir) if "_DF_" not in f]
    stats = defaultdict(int)
    processed = []

    for doc_id in tqdm(doc_ids, desc=lang):
        source_file = os.path.join(source_dir, doc_id + ".xml")
        ere_file = os.path.join(ere_dir, doc_id + ".rich_ere.xml")
        annotation_file = os.path.join(annotation_dir, doc_id + ".best.xml")

        source_xml = ET.parse(source_file).getroot()
        ere_xml = ET.parse(ere_file).getroot()
        annotation_xml = ET.parse(annotation_file).getroot()

        ent_pos_sentiment_xmls = annotation_xml.findall(".//entity/sentiments/sentiment[@polarity='pos']")
        ent_neg_sentiment_xmls = annotation_xml.findall(".//entity/sentiments/sentiment[@polarity='neg']")
        evt_pos_sentiment_xmls = annotation_xml.findall(".//event/sentiments/sentiment[@polarity='pos']")
        evt_neg_sentiment_xmls = annotation_xml.findall(".//event/sentiments/sentiment[@polarity='neg']")

        source_types = []
        for xml in ent_pos_sentiment_xmls + ent_neg_sentiment_xmls + evt_pos_sentiment_xmls + evt_neg_sentiment_xmls:
            source_types.append(xml.find("source").attrib["type"])

        stats["00_n_pos_sentiment"] += len(ent_pos_sentiment_xmls) + len(evt_pos_sentiment_xmls)
        stats["01_n_neg_sentiment"] += len(ent_neg_sentiment_xmls) + len(evt_neg_sentiment_xmls)
        stats["02_n_ent_sentiment"] += len(ent_pos_sentiment_xmls) + len(ent_neg_sentiment_xmls)
        stats["03_n_evt_sentiment"] += len(evt_pos_sentiment_xmls) + len(evt_neg_sentiment_xmls)
        stats["04_n_author-source_sentiment"] += sum([s == "author" for s in source_types])
        stats["05_n_other-source_sentiment"] += sum([s == "other" for s in source_types])
        stats["06_n_entity-source_sentiment"] += sum([s == "mention" for s in source_types])

        source = open(source_file, encoding="utf-8").read()
        headline = source_xml.find("HEADLINE")
        author = source_xml.find("AUTHOR")
        text = source_xml.find("TEXT")
        paras = text.findall("P")

        source_texts = []

        if headline is not None and headline.text is not None:
            source_texts.append({"type": "headline", "text": headline.text.strip()})

        if author is not None and author.text is not None:
            source_texts.append({"type": "author", "text": author.text.strip()})

        if paras:
            for para in paras:
                source_texts.append({"type": "body", "text": para.text.strip()})
        else:
            source_texts.append({"type": "body", "text": text.text.strip()})
        
        tokenized_texts = []

        for i in range(len(source_texts)):
            text = source_texts[i]["text"]
            pattern = "({})|({})".format(re.escape(text), re.escape(html.escape(text)))
            for match in re.finditer(pattern, source):
                offset = match.span()[0]
                if i == 0 or source_texts[i - 1]["source_offset"] is None or offset > source_texts[i - 1]["source_offset"] + len(source_texts[i - 1]["text"]):
                    source_texts[i]["source_offset"] = offset
                    doc = spacy_model(text)
                    for sent in doc.sents:
                        tokens = [t.text for t in sent]
                        offsets = [offset + t.idx for t in sent]
                        tokenized_texts.append({"tokens": tokens, "offsets": offsets, "source_offset": offset, "type": source_texts[i]["type"], "text": sent.text})
                    break
            else:
                source_texts[i]["source_offset"] = None
                tokenized_texts.append(source_texts[i])
                stats["07_n_source-text-element_offset-not-found"] += 1

        annotations = []

        for target_xml in annotation_xml.findall(".//sentiment_annotations/entities/entity") + annotation_xml.findall(".//sentiment_annotations/events/event"):
            target = {"type": target_xml.tag, "ere_id": target_xml.attrib["ere_id"]}
            if target["type"] == "entity":
                target_entity_mention = ere_xml.find(".//entity_mention[@id='{}']".format(target["ere_id"]))
                target.update({"offset": int(target_entity_mention.attrib["offset"]), "length": int(target_entity_mention.attrib["length"])})
            else:
                target_event_trigger = ere_xml.find(".//event_mention[@id='{}']/trigger".format(target["ere_id"]))
                target.update({"offset": int(target_event_trigger.attrib["offset"]), "length": int(target_event_trigger.attrib["length"])})
            target["index"] = find_index(target["offset"], target["length"], tokenized_texts)

            
            for sentiment in target_xml.findall("./sentiments/sentiment[@polarity='pos']") + target_xml.findall("./sentiments/sentiment[@polarity='neg']"):
                polarity = sentiment.attrib["polarity"]
                source_xml = sentiment.find("source")
                source = {"type": source_xml.attrib["type"]}
                
                if source["type"] == "mention":
                    source["ere_id"] = source_xml.attrib["ere_id"]
                    source_entity_mention = ere_xml.find(".//entity_mention[@id='{}']".format(source["ere_id"]))
                    source.update({"offset": int(source_entity_mention.attrib["offset"]), "length": int(source_entity_mention.attrib["length"])})
                    parent_xml = ere_xml.find(".//entity_mention[@id='{}']/..".format(source["ere_id"]))
                    source["entity_ere_id"] = parent_xml.attrib["id"]
                    source["index"] = find_index(source["offset"], source["length"], tokenized_texts)
                    source["valid_nearest_index"] = []

                    if target["index"] is not None and target["index"][2] != -1:
                        for entity_mention in ere_xml.findall(".//entity[@id='{}']/entity_mention".format(source["entity_ere_id"])):
                            entity_mention_offset = int(entity_mention.attrib["offset"])                    
                            entity_mention_length = int(entity_mention.attrib["length"])
                            entity_mention_index = find_index(entity_mention_offset, entity_mention_length, tokenized_texts)
                            if entity_mention_index is not None and entity_mention_index[2] != -1:
                                source["valid_nearest_index"].append(entity_mention_index)

                        source["valid_nearest_index"] = sorted(source["valid_nearest_index"], key = lambda index: abs(index[0] - target["index"][0]))
                        if source["valid_nearest_index"]:
                            source["valid_nearest_index"] = source["valid_nearest_index"][0]
                        else:
                            source["valid_nearest_index"] = None
                    else:
                        source["valid_nearest_index"] = None

                if target["index"] is None:
                    stats["08_n_sentiment_target-offset-not-found"] += 1
                elif target["index"][2] == -1:
                    stats["09_n_sentiment_target-cross-sentence-boundary"] += 1
                else:
                    stats["10_n_sentiment_target-offset-found"] += 1
                    if source["type"] == "author":
                        stats["11_n_sentiment_target-offset-found_author-source"] += 1
                    elif source["type"] == "other":
                        stats["12_n_sentiment_target-offset-found_other-source"] += 1
                    else:
                        stats["13_n_sentiment_target-offset-found_entity-source"] += 1
                        if source["index"] is None:
                            stats["14_n_sentiment_target-offset-found_entity-source_source-offset-not-found"] += 1
                        elif source["index"][2] == -1:
                            stats["15_n_sentiment_target-offset-found_entity-source_source-cross-sentence-boundary"] += 1
                        elif source["index"][0] != target["index"][0]:
                            stats["16_n_sentiment_target-offset-found_entity-source_source-and-target-are-in-different-sentence"] += 1
                        else:
                            stats["17_n_sentiment_target-offset-found_entity-source_source-and-target-are-in-same-sentence"] += 1
                        if source["valid_nearest_index"] is not None and source["valid_nearest_index"][2] != -1 and source["valid_nearest_index"][0] == target["index"][0]:
                            stats["18_n_sentiment_target-offset-found_entity-source_alternative-source-and-target-are-in-same-sentence"] += 1

                annotations.append({"target": target, "source": source, "polarity": polarity})
        
        processed.append({"doc_id": doc_id, "tokenized_texts": tokenized_texts, "annotations": annotations})
    
    processed = {"processed": processed, "stats": stats}
    processed_file = os.path.join(results_dir, lang + ".json")
    json.dump(processed, open(processed_file, "w", encoding="utf-8"), indent=2, sort_keys=True)