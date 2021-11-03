# author : Sabyasachee Baruah

import json
import os
import config
from collections import Counter

mpqa2_docids = open(os.path.join(config.MPQA2_FOLDER, "doclist.attitudeSubset")).read().splitlines()
mpqa3_docids = open(os.path.join(config.MPQA3_FOLDER, "doclist")).read().splitlines()

mpqa2_attitudetypes = []
mpqa3_attitudetypes = []

mpqa2_holdertypes = []
mpqa3_holdertypes = []

for docid in mpqa2_docids:
    file = os.path.join(config.RESULTS_FOLDER, "mpqa2-processed", docid, "tokenized.json")
    doc = json.load(open(file))
    for sentence in doc:
        for dse in sentence["dse"]:
            mpqa2_attitudetypes.append(dse["attitude-type"])
            mpqa2_holdertypes.append(dse["holder-type"])

for docid in mpqa3_docids:
    file = os.path.join(config.RESULTS_FOLDER, "mpqa3-processed", docid, "tokenized.json")
    doc = json.load(open(file))
    for sentence in doc:
        for dse in sentence["dse"]:
            if dse["target-type"] == "span":
                mpqa3_attitudetypes.append(dse["attitude-type"])
                mpqa3_holdertypes.append(dse["holder-type"])

print(Counter(mpqa2_attitudetypes))
print(Counter(mpqa3_attitudetypes))

print(Counter(mpqa2_holdertypes))
print(Counter(mpqa3_holdertypes))