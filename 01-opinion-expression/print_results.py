import os
import re
import config
import argparse
import numpy as np
from collections import defaultdict

def parse_log_line(line):
    p, r, f = re.search("precision = (0\.\d+), recall = (0\.\d+), F1 = (0\.\d+)", line).groups()
    return np.array([float(p), float(r), float(f)])

def print_results(model_name):
    model_folder = os.path.join(config.MODEL_FOLDER, model_name)
    results = np.zeros((2, 2, 2, 3), dtype=float)
    headers = [["dev", "test"], ["neg", "pos"], ["binary", "proportional"], ["precision", "recall", "F1"]]

    for fold in range(4):
        logfile = os.path.join(model_folder, f"fold{fold}", "log.txt")
        lines = open(logfile).read().split("\n")
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    line = lines[26 + 7*i + 2*j + k]
                    results[i, j, k] += parse_log_line(line)
    
    results = results/4

    for i in range(2):
        print(headers[0][i])
        for j in range(2):
            print("\t", headers[1][j])
            for k in range(2):
                print("\t\t", headers[2][k])
                for l in range(3):
                    print("\t\t\t{:10s} = {:.3f}".format(headers[3][l], results[i, j, k, l]))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="print dev/test performance of given model")
    parser.add_argument("-m", type=str, help="model-name")
    args = parser.parse_args()
    
    model_name = args.m

    print_results(model_name)