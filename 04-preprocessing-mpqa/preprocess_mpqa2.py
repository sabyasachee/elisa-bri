# author = Sabyasachee Baruah
# This script processes the MPQA 2.0

from collections import defaultdict, Counter
import os
import re
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm
import json
import numpy as np
import config

def parse_annotation_line(text: str) -> Dict[str, Any]:
    parts = text.split("\t")
    id = parts[0].strip()
    start, end = parts[1].strip().split(",")
    start = int(start)
    end = int(end)
    type = parts[3].strip()
    attributes = {}
    if len(parts) > 4:
        attribute_str = parts[4].strip()
        pattern = '[^=]+="[^"]+"'
        for match in re.findall(pattern, attribute_str):
            key, values = match.strip().split("=")
            values = values.strip('"').split(",")
            values = list(map(str.strip, values))
            values = list(filter(lambda x: x != "none", values))
            match = re.search("[\w-]+", key)
            if len(values) > 0 and match is not None:
                key = match.group(0)
                attributes[key] = values
    parsed_annotation_line = {
        "id": id,
        "span": {
            "start": start,
            "end": end
        },
        "type": type,
        "attributes": attributes
    }
    return parsed_annotation_line

def parse_sentence_annotation_line(text: str) -> Dict[str, int]:
    parts = text.split("\t")
    start, end = parts[1].strip().split(",")
    start = int(start)
    end = int(end)
    return {
        "start": start,
        "end": end
    }

def remove_subset_spans(spans: List[Dict[str, int]]) -> List[Dict[str, int]]:
    spans = sorted(spans, key = lambda x: (x["start"], -x["end"]))
    new_spans = []
    for i, span in enumerate(spans):
        if i == 0 or new_spans[-1]["end"] <= span["start"]:
            new_spans.append(span)
    return new_spans

def find_sentence_index(span: Dict[str, int], sentence_spans: List[Dict[str, int]]) -> Union[None, int]:
    for i, sentence_span in enumerate(sentence_spans):
        if sentence_span["start"] <= span["start"] and span["end"] <= sentence_span["end"]:
            return i

def find_attribute(key: str, attributes: Dict[str, List[str]], attribute_type: str = "list") -> Union[None, str, List[str]]:
    assert attribute_type in ["list", "first", "last"]
    if key in attributes:
        values = attributes[key]
        if len(values) > 0:
            if attribute_type == "first":
                return values[0]
            elif attribute_type == "last":
                return values[-1]
            else:
                return values
    if attribute_type == "list":
        return []

def constrain_values(value: Union[str, List[str]], allowable_values: List[str]) -> Union[None, str, List[str]]:
    if isinstance(value, list):
        new_values = []
        for v in value:
            if v in allowable_values:
                new_values.append(v)
        return new_values
    else:
        if value in allowable_values:
            return value

def remove_redundancies_in_nested_source(nested_source: List[str]) -> List[str]:
    new_nested_source = []
    i = 0
    while i < len(nested_source):
        j = i + 1
        while j < len(nested_source) and nested_source[j] == nested_source[i]:
            j += 1
        new_nested_source.append(nested_source[i])
        i = j
    return new_nested_source

def parse_agent(parsed_annotation: Dict[str, Any]) -> Dict[str, Any]:
    agent = {}
    attrs = parsed_annotation["attributes"]
    agent["span"] = parsed_annotation["span"]
    agent["id"] = find_attribute("id", attrs, attribute_type="last")
    agent["nested-source"] = find_attribute("nested-source", attrs)
    agent["non-redundant-nested-source"] = remove_redundancies_in_nested_source(agent["nested-source"])

    x = find_attribute("agent-uncertain", attrs, attribute_type="first")
    agent["agent-uncertain"] = constrain_values(x, ["somewhat-uncertain", "very-uncertain"])
    
    return agent

def parse_direct_subjective(parsed_annotation: Dict[str, Any]) -> Dict[str, Any]:
    dse = {}
    attrs = parsed_annotation["attributes"]
    dse["span"] = parsed_annotation["span"]
    dse["attitude-link"] = find_attribute("attitude-link", attrs)
    dse["nested-source"] = find_attribute("nested-source", attrs)
    dse["non-redundant-nested-source"] = remove_redundancies_in_nested_source(dse["nested-source"])
    
    x = find_attribute("intensity", attrs, attribute_type="first")
    dse["intensity"] = constrain_values(x, ["neutral", "low", "medium", "high", "extreme"])

    dse["implicit"] = find_attribute("implicit", attrs, attribute_type="first") == "true"
    
    x = find_attribute("expression-intensity", attrs, attribute_type="first")
    dse["expression-intensity"] = constrain_values(x, ["neutral", "low", "medium", "high", "extreme"])

    x = find_attribute("polarity", attrs, attribute_type="first")
    dse["polarity"] = constrain_values(x, ["neutral", "positive", "negative", "both", "uncertain-both", \
    "uncertain-neutral", "uncertain-positive", "uncertain-negative"])

    x = find_attribute("insubstantial", attrs)
    x = list(map(str.lower, x))
    dse["insubstantial"] = constrain_values(x, ["c1","c2","c3"])

    x = find_attribute("annotation-uncertain", attrs, attribute_type="first")
    dse["annotation-uncertain"] = constrain_values(x, ["somewhat-uncertain", "very-uncertain"])

    x = find_attribute("subjective-uncertain", attrs, attribute_type="first")
    dse["subjective-uncertain"] = constrain_values(x, ["somewhat-uncertain", "very-uncertain"])

    return dse

def parse_expressive_subjective(parsed_annotation: Dict[str, Any]) -> Dict[str, Any]:
    ese = {}
    attrs = parsed_annotation["attributes"]
    ese["span"] = parsed_annotation["span"]
    ese["nested-source"] = find_attribute("nested-source", attrs)
    ese["non-redundant-nested-source"] = remove_redundancies_in_nested_source(ese["nested-source"])

    x = find_attribute("intensity", attrs, attribute_type="first")
    ese["intensity"] = constrain_values(x, ["low", "medium", "high", "extreme"])

    x = find_attribute("polarity", attrs, attribute_type="first")
    ese["polarity"] = constrain_values(x, ["neutral", "positive", "negative", "both", "uncertain-both", \
    "uncertain-neutral", "uncertain-positive", "uncertain-negative"])

    x = find_attribute("es-uncertain", attrs, attribute_type="first")
    ese["es-uncertain"] = constrain_values(x, ["somewhat-uncertain", "very-uncertain"])

    x = find_attribute("nested-source-uncertain", attrs, attribute_type="first")
    ese["nested-source-uncertain"] = constrain_values(x, ["somewhat-uncertain", "very-uncertain"])

    return ese

def parse_attitude(parsed_annotation: Dict[str, Any]) -> Dict[str, Any]:
    attitude = {}
    attrs = parsed_annotation["attributes"]
    attitude["id"] = find_attribute("id", attrs, attribute_type="first")
    attitude["span"] = parsed_annotation["span"]
    attitude["target-link"] = find_attribute("target-link", attrs)
    
    x = find_attribute("attitude-type", attrs, attribute_type="first")
    attitude["attitude-type"] = constrain_values(x, ["sentiment-neg", "sentiment-pos", "arguing-neg", "arguing-pos", \
        "agree-neg", "agree-pos", "intention-neg", "intention-pos", "speculation", "other-attitude"])

    x = find_attribute("intensity", attrs, attribute_type="first")
    attitude["intensity"] = constrain_values(x, ["low", "low-medium", "medium", "medium-high", "high", "high-extreme", "extreme"])

    x = find_attribute("attitude-uncertain", attrs, attribute_type="first")
    attitude["attitude-uncertain"] = constrain_values(x, ["somewhat-uncertain", "very-uncertain"])

    attitude["inferred"] = find_attribute("inferred", attrs, attribute_type="first") == "yes"
    attitude["repetition"] = find_attribute("repetition", attrs, attribute_type="first") == "yes"
    attitude["contrast"] = find_attribute("contrast", attrs, attribute_type="first") == "yes"
    attitude["sarcastic"] = find_attribute("sarcastic", attrs, attribute_type="first") == "yes"

    return attitude

def parse_target(parsed_annotation: Dict[str, Any]) -> Dict[str, Any]:
    target = {}
    attrs = parsed_annotation["attributes"]
    target["id"] = find_attribute("id", attrs, attribute_type="first")
    target["span"] = parsed_annotation["span"]
    
    x = find_attribute("target-uncertain", attrs, attribute_type="first")
    target["target-uncertain"] = constrain_values(x, ["somewhat-uncertain", "very-uncertain"])

    return target

def nested_source_is_writer(nested_source: List[str]) -> bool:
    nested_source = remove_redundancies_in_nested_source(nested_source)
    return nested_source == ["w"]

def nested_source_is_implicit(nested_source: List[str]) -> bool:
    nested_source = remove_redundancies_in_nested_source(nested_source)
    return nested_source in [["w", "implicit"], ["implicit"]]

def find_closest_agent(key, sentence_index, span, agents, use_non_redundant_nested_source=False, use_id=False) -> Union[None, Dict[str, Any]]:
    closest_agent = None
    closest_distance = (np.inf, np.inf)
    agent_key = "non-redundant-nested-source" if use_non_redundant_nested_source else "nested-source"
    for agent in agents:
        if (use_id and agent["id"] is not None and agent["id"] == key) or (agent[agent_key] == key):
            distance = (abs(agent["sentence-index"] - sentence_index), abs((agent["span"]["start"] + agent["span"]["end"])/2 - (span["start"] + span["end"])/2))
            if distance < closest_distance:
                closest_distance = distance
                closest_agent = agent
    return closest_agent

def match_nested_source_to_agents(subjective: Dict[str, Any], agents: List[Dict[str, Any]]) -> Union[None, Tuple[Dict[str, Any], str]]:
    valid_agents = list(filter(lambda agent: agent["sentence-index"] is not None, agents))
    nested_source = subjective["nested-source"]
    non_redundant_nested_source = subjective["non-redundant-nested-source"]
    sentence_index = subjective["sentence-index"]
    span = subjective["span"]

    if sentence_index is not None:
        for agent_non_redundancy in [False, True]:
            d = "1" if agent_non_redundancy else "0"
            agent = find_closest_agent(nested_source, sentence_index, span, valid_agents, use_non_redundant_nested_source=agent_non_redundancy)
            if agent is not None:
                return agent, d + "0"
            
            agent = find_closest_agent(non_redundant_nested_source, sentence_index, span, valid_agents, use_non_redundant_nested_source=agent_non_redundancy)
            if agent is not None:
                return agent, d + "1"

            if nested_source[-1] == "implicit" and len(nested_source) > 1:
                key = nested_source[:-1]
                agent = find_closest_agent(key, sentence_index, span, valid_agents, use_non_redundant_nested_source=agent_non_redundancy)
                if agent is not None:
                    return agent, d + "2"

            if nested_source[0] != "w":
                key = ["w"] + nested_source
                agent = find_closest_agent(key, sentence_index, span, valid_agents, use_non_redundant_nested_source=agent_non_redundancy)
                if agent is not None:
                    return agent, d + "3"

            if nested_source[0] == "w" and len(nested_source) > 1:
                key = nested_source[1:]
                agent = find_closest_agent(key, sentence_index, span, valid_agents, use_non_redundant_nested_source=agent_non_redundancy)
                if agent is not None:
                    return agent, d + "4"

            if nested_source[-1] != "implicit":
                key = nested_source[-1]
                agent = find_closest_agent(key, sentence_index, span, valid_agents, use_id=True)
                if agent is not None:
                    return agent, "20"
    else:
        agent = find_closest_agent(nested_source, np.inf, span, valid_agents)
        if agent is not None:
            return agent, "30"

def preprocess_mpqa2(mpqa2_folder: str, results_folder: str):
    doc_ids_file = os.path.join(mpqa2_folder, "doclist.attitudeSubset")
    doc_ids = open(doc_ids_file).read().strip().split("\n")
    mpqa_stats = defaultdict(int)
    mpqa_dse_matched_source_stats = defaultdict(int)
    mpqa_ese_matched_source_stats = defaultdict(int)

    for doc_id in tqdm(doc_ids):
        doc_file = os.path.join(mpqa2_folder, "docs", doc_id)
        annotation_file = os.path.join(mpqa2_folder, "man_anns", doc_id, "gateman.mpqa.lre.2.0")
        sentences_file = os.path.join(mpqa2_folder, "man_anns", doc_id, "gatesentences.mpqa.2.0")
        
        stats = defaultdict(int)
        distance_between_dse_and_source = []
        distance_between_ese_and_source = []
        distance_between_dse_and_target = []

        text = open(doc_file).read()

        annotation_lines = open(annotation_file).read().strip().split("\n")
        annotation_lines = list(filter(lambda line: not line.startswith("#"), annotation_lines))
        parsed_annotations = list(map(parse_annotation_line, annotation_lines))

        sentence_annotation_lines = open(sentences_file).read().strip().split("\n")
        sentence_annotation_lines = list(filter(lambda line: not line.startswith("#"), sentence_annotation_lines))
        parsed_sentence_annotations = list(map(parse_sentence_annotation_line, sentence_annotation_lines))
        parsed_sentence_annotations = remove_subset_spans(parsed_sentence_annotations)
        parsed_sentence_annotations = sorted(parsed_sentence_annotations, key = lambda x: (x["start"], -x["end"]))

        sentences: List[Dict[str, Any]] = []
        for parsed_sentence_annotation in parsed_sentence_annotations:
            sentence_text = text[parsed_sentence_annotation["start"]: parsed_sentence_annotation["end"]]
            sentences.append({
                "span": parsed_sentence_annotation,
                "text": sentence_text
            })

        agents, direct_subjectives, expressive_subjectives, attitudes, targets = [], [], [], [], []
        sentence_spans = [sentence["span"] for sentence in sentences]

        for parsed_annotation in parsed_annotations:
            sentence_index = find_sentence_index(parsed_annotation["span"], sentence_spans)
            
            if parsed_annotation["type"] == "GATE_agent":
                agent = parse_agent(parsed_annotation)
                agent["sentence-index"] = sentence_index
                agents.append(agent)
            
            elif parsed_annotation["type"] == "GATE_direct-subjective":
                dse = parse_direct_subjective(parsed_annotation)
                dse["sentence-index"] = sentence_index
                direct_subjectives.append(dse)

            elif parsed_annotation["type"] == "GATE_expressive-subjectivity":
                ese = parse_expressive_subjective(parsed_annotation)
                ese["sentence-index"] = sentence_index
                expressive_subjectives.append(ese)

            elif parsed_annotation["type"] == "GATE_attitude":
                att = parse_attitude(parsed_annotation)
                att["sentence-index"] = sentence_index
                attitudes.append(att)

            elif parsed_annotation["type"] == "GATE_target":
                tgt = parse_target(parsed_annotation)
                tgt["sentence-index"] = sentence_index
                targets.append(tgt)
        
        for sname, subjectives, distance_between_subj_and_source in [("dse", direct_subjectives, distance_between_dse_and_source), \
            ("ese", expressive_subjectives, distance_between_ese_and_source)]:
            d = "0" if sname == "dse" else "2"
            for subjective in subjectives:
                stats[f"{d}0_n_{sname}"] += 1
                subjective["matched-source"] = None
                subjective["matched-source-type"] = None
                if len(subjective["nested-source"]) > 0:
                    stats[f"{d}1_n_{sname}_nested-source-present"] += 1
                    if nested_source_is_implicit(subjective["nested-source"]):
                        subjective["matched-source"] = "implicit"
                        stats[f"{d}3_n_{sname}_nested-source-present_implicit-source"] += 1
                    elif nested_source_is_writer(subjective["nested-source"]):
                        subjective["matched-source"] = "writer"
                        stats[f"{d}4_n_{sname}_nested-source-present_writer-source"] += 1
                    else:
                        ret = match_nested_source_to_agents(subjective, agents)
                        if ret is not None:
                            stats[f"{d}5_n_{sname}_nested-source-present_span-source"] += 1
                            subjective["matched-source"] = ret[0]
                            subjective["matched-source-type"] = ret[1]
                            if subjective["sentence-index"] is not None:
                                distance = abs(ret[0]["sentence-index"] - subjective["sentence-index"])
                                distance_between_subj_and_source.append(distance)
                        else:
                            stats[f"{d}6_n_{sname}_nested-source-present_source-not-found"] += 1
                else:
                    stats[f"{d}2_n_{sname}_nested-source-absent"] += 1

        for dse in direct_subjectives:
            dse["matched-attitudes"] = []
            for att_id in dse["attitude-link"]:
                stats["10_n_dse-attitude"] += 1
                for att in attitudes:
                    if att["id"] is not None and att["id"] == att_id:
                        stats["11_n_dse-attitude_attitude-found"] += 1
                        att["matched-targets"] = []
                        for tgt_id in att["target-link"]:
                            stats["13_n_dse-attitude_attitude-found_attitude-target"] += 1
                            for tgt in targets:
                                if tgt["id"] is not None and tgt["id"] == tgt_id:
                                    stats["14_n_dse-attitude_attitude-found_attitude-target_target-found"] += 1
                                    if tgt["sentence-index"] is not None:
                                        stats["16_n_dse-attitude_attitude-found_attitude-target_target-found_target-span-found"] += 1
                                        att["matched-targets"].append(tgt)
                                        if dse["sentence-index"] is not None:
                                            distance = abs(dse["sentence-index"] - tgt["sentence-index"])
                                            distance_between_dse_and_target.append(distance)
                                    else:
                                        stats["17_n_dse-attitude_attitude-found_attitude-target_target-found_target-span-not-found"] += 1
                                    break
                            else:
                                stats["15_n_dse-attitude_attitude-found_attitude-target_target-not-found"] += 1
                        dse["matched-attitudes"].append(att)
                        break
                else:
                    stats["12_n_dse-attitude_attitude-not-found"] += 1

        stats["30_distance_between_dse_and_source"] = dict(Counter(distance_between_dse_and_source))
        stats["31_distance_between_ese_and_source"] = dict(Counter(distance_between_ese_and_source))
        stats["32_distance_between_dse_and_target"] = dict(Counter(distance_between_dse_and_target))
    
        processed = {
            "text": text,
            "sentences": sentences,
            "dse": direct_subjectives,
            "ese": expressive_subjectives,
            "stats": stats
        }

        folder = os.path.join(results_folder, "mpqa2-processed", doc_id)
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, "processed.json")
        json.dump(processed, open(file, "w"), indent=2, sort_keys=True)
    
        for key, value in stats.items():
            if type(value) == int:
                mpqa_stats[key] += value
            else:
                if key not in mpqa_stats:
                    mpqa_stats[key] = defaultdict(int)
                for key2, value2 in value.items():
                    mpqa_stats[key][key2] += value2
    
        for subjectives, matched_source_stats in [(direct_subjectives, mpqa_dse_matched_source_stats), \
            (expressive_subjectives, mpqa_ese_matched_source_stats)]:
            for subjective in subjectives:
                if subjective["matched-source-type"] is not None:
                    matched_source_stats[subjective["matched-source-type"]] += 1
        

    mpqa_stats["33_dse_matched_source_type"] = mpqa_dse_matched_source_stats
    mpqa_stats["34_ese_matched_source_type"] = mpqa_ese_matched_source_stats
    stats_file = os.path.join(results_folder, "mpqa2-processed/stats.json")
    json.dump(mpqa_stats, open(stats_file, "w"), indent=2, sort_keys=True)

if __name__=="__main__":
    preprocess_mpqa2(config.MPQA2_FOLDER, config.RESULTS_FOLDER)