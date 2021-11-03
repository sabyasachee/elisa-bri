# author : Sabyasachee

import torch
import config
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from collections import defaultdict
from models.RoleLabeler import RoleLabeler
from transformers import get_linear_schedule_with_warmup

pd.set_option("precision", 2)

def overlap(x, y):
    return int(x[0] < y[1] and y[0] < x[1])

def proportion(x, y):
    return max(0, min(x[1] - x[0], y[1] - y[0], x[1] - y[0], y[1] - x[0]))/(x[1] - x[0])

def exact(x, y):
    return int(x == y)

def zmax(T):
    try:
        return max(T)
    except:
        return 0

def zmean(T):
    try:
        return np.mean(T)
    except:
        return 0

def evaluate(sentenceid_to_pred_and_true, true_key="true", pred_key="pred"):
    '''
    sentenceid_to_pred_and_true is a dictionary of the form:
    {sentenceid: {"true": set, "pred": set}}
    true and pred set is a set of 5-tuples
    Each 5-tuple is the concatenation of the holder span, target span, and opinion label (0, 1, or 2)

    We evaluate 4 subtasks:
        1. holder prediction (span)
        2. target prediction (span)
        3. frame prediction w/o attitude type (span pair)
        4. frame prediction w attitude type (span pair + label)
    
    let x, y, u, v be spans. p and q be opinion labels.
    let T be true spans or span pairs or span pairs + labels
    let P be predicted spans or span pairs or span pairs + labels

    O(x, y) = 1 if x overlaps y, else 0
    L(x, y) = length of overlap between x and y
    E(x, y) = 1 if x == y, else 0
    A(p, q) = 1 if p equals q, else 0
    S(A, B, s) = (1/|A|) * (SUM_(x in A) MAX_(y in B) s(x, y))

    We use 3 metrics:
        1. Binary: 
            b1(x, y) = O(x, y)
            b2(x, y, u, v) = O(x, u) * O(y, v)
            b3(x, y, p, u, v, q) = b2(x, y, u, v) * A(p, q)

            span case:              recall = S(T, P, b1), precision = S(P, T, b1)
            span pair case:         recall = S(T, P, b2), precision = S(P, T, b2)
            span pair case + label: recall = S(T, P, b3), precision = S(P, T, b3)

        2. Proportional:
            p1(x, y) = L(x, y)/|x|
            p2(x, y, u, v) = 1/2 * O(x, u) * O(y, v) * (L(x, u)/|x| + L(y, v)/|y|)
            p3(x, y, p, u, v, q) = p2(x, y, u, v) * A(p, q)

            span case:              recall = S(T, P, p1), precision = S(P, T, p1)
            span pair case:         recall = S(T, P, p2), precision = S(P, T, p2)
            span pair case + label: recall = S(T, P, p3), precision = S(P, T, p3)

        3. Exact:
            e1(x, y) = E(x, y)
            e2(x, y, u, v) = E(x, u) * E(y, v)
            e3(x, y, p, u, v, q) = e2(x, y, u, v) * A(p, q)

            span case:              recall = S(T, P, e1), precision = S(P, T, e1)
            span pair case:         recall = S(T, P, e2), precision = S(P, T, e2)
            span pair case + label: recall = S(T, P, e3), precision = S(P, T, e3)

    The function returns results dataframe.
    The index is subtasks: [holder, target, frame-without-attitude, frame-with-attitude]
    The column is metrics: [binP, binR, binF, proP, proR, proF, excP, excR, excF]
    The shape is 4 x 9
    '''

    hol_binR, tgt_binR, frm_binR, fra_binR = [], [], [], []
    hol_binP, tgt_binP, frm_binP, fra_binP = [], [], [], []
    hol_proR, tgt_proR, frm_proR, fra_proR = [], [], [], []
    hol_proP, tgt_proP, frm_proP, fra_proP = [], [], [], []
    hol_excR, tgt_excR, frm_excR, fra_excR = [], [], [], []
    hol_excP, tgt_excP, frm_excP, fra_excP = [], [], [], []

    for _, pred_and_true in sentenceid_to_pred_and_true.items():
        pred_holders = set([(example[0], example[1]) for example in pred_and_true[pred_key]])
        pred_targets = set([(example[2], example[3]) for example in pred_and_true[pred_key]])
        true_holders = set([(example[0], example[1]) for example in pred_and_true[true_key]])
        true_targets = set([(example[2], example[3]) for example in pred_and_true[true_key]])
        pred_frames = set([((example[0], example[1]), (example[2], example[3]), example[4]) for example in pred_and_true[pred_key]])
        true_frames = set([((example[0], example[1]), (example[2], example[3]), example[4]) for example in pred_and_true[true_key]])

        # holder recall
        for x in true_holders:
            hol_binR.append(zmax(overlap(x, y) for y in pred_holders))
            hol_proR.append(zmax(proportion(x, y) for y in pred_holders))
            hol_excR.append(zmax(exact(x, y) for y in pred_holders))
        
        # holder precision
        for y in pred_holders:
            hol_binP.append(zmax(overlap(y, x) for x in true_holders))
            hol_proP.append(zmax(proportion(y, x) for x in true_holders))
            hol_excP.append(zmax(exact(y, x) for x in true_holders))
        
        # target recall
        for x in true_targets:
            tgt_binR.append(zmax(overlap(x, y) for y in pred_targets))
            tgt_proR.append(zmax(proportion(x, y) for y in pred_targets))
            tgt_excR.append(zmax(exact(x, y) for y in pred_targets))
        
        # target precision
        for y in pred_targets:
            tgt_binP.append(zmax(overlap(y, x) for x in true_targets))
            tgt_proP.append(zmax(proportion(y, x) for x in true_targets))
            tgt_excP.append(zmax(exact(y, x) for x in true_targets))
        
        # frame-without-attitude recall
        for x, y, _ in true_frames:
            frm_binR.append(zmax(overlap(x, u) * overlap(y, v) for u, v, _ in pred_frames))
            frm_proR.append(zmax(0.5 * overlap(x, u) * overlap(y, v) * (proportion(x, u) + proportion(y, v)) for u, v, _ in pred_frames))
            frm_excR.append(zmax(exact(x, u) * exact(y, v) for u, v, _ in pred_frames))

        # frame-without-attitude precision
        for u, v, _ in pred_frames:
            frm_binP.append(zmax(overlap(u, x) * overlap(y, v) for x, y, _ in true_frames))
            frm_proP.append(zmax(0.5 * overlap(u, x) * overlap(v, y) * (proportion(u, x) + proportion(v, y)) for x, y, _ in true_frames))
            frm_excP.append(zmax(exact(u, x) * exact(v, y) for x, y, _ in true_frames))

        # frame-with-attitude recall
        for x, y, p in true_frames:
            fra_binR.append(zmax(overlap(x, u) * overlap(y, v) * (p == q) for u, v, q in pred_frames))
            fra_proR.append(zmax(0.5 * overlap(x, u) * overlap(y, v) * (proportion(x, u) + proportion(y, v)) * (p == q) for u, v, q in pred_frames))
            fra_excR.append(zmax(exact(x, u) * exact(y, v) * (p == q) for u, v, q in pred_frames))

        # frame-with-attitude precision
        for u, v, q in pred_frames:
            fra_binP.append(zmax(overlap(u, x) * overlap(y, v) * (p == q) for x, y, p in true_frames))
            fra_proP.append(zmax(0.5 * overlap(u, x) * overlap(v, y) * (proportion(u, x) + proportion(v, y)) * (p == q) for x, y, p in true_frames))
            fra_excP.append(zmax(exact(u, x) * exact(v, y) * (p == q) for x, y, p in true_frames))
        
    data = np.zeros((4, 6))
    data[0] = [zmean(hol_binP), zmean(hol_binR), zmean(hol_proP), zmean(hol_proR), zmean(hol_excP), zmean(hol_excR)]
    data[1] = [zmean(tgt_binP), zmean(tgt_binR), zmean(tgt_proP), zmean(tgt_proR), zmean(tgt_excP), zmean(tgt_excR)]
    data[2] = [zmean(frm_binP), zmean(frm_binR), zmean(frm_proP), zmean(frm_proR), zmean(frm_excP), zmean(frm_excR)]
    data[3] = [zmean(fra_binP), zmean(fra_binR), zmean(fra_proP), zmean(fra_proR), zmean(fra_excP), zmean(fra_excR)]

    df = pd.DataFrame(data, index=["holder","target","frame-without-attitude","frame-with-attitude"], columns=["binP","binR","proP","proR","excP","excR"])
    df["binF"] = 2 * df["binP"] * df["binR"] / (df["binP"] + df["binR"] + 1e-23)
    df["proF"] = 2 * df["proP"] * df["proR"] / (df["proP"] + df["proR"] + 1e-23)
    df["excF"] = 2 * df["excP"] * df["excR"] / (df["excP"] + df["excR"] + 1e-23)
    df *= 100

    return df

def inference(dataset, model):
    '''
    Run model inference on dataset to return predictions.
    The predictions are in the form of a dictionary.
    predictions = {sentenceid: {"true": set, "pred": set}}.
    
    The true and pred set is a set of tuples.
    Each tuple is a 5-tuple. 
    The first two elements is the holder span, the next two elements is the target span, 
    and the last element is the opinion label.
    '''

    sentenceid_to_pred_and_true = defaultdict(lambda: defaultdict(set))
    model.eval()

    with torch.no_grad():
        for X_wordid, X_mask, Y in tqdm(dataset.data, desc="inference", total=len(dataset)):
            
            Y = Y.cpu().numpy()
            unique_index = []
            batch_dataset_sentenceids = set()
            batch_sentenceid_to_dataset_sentenceid = {}

            for i in range(len(Y)):
                
                if Y[i, 5] != -1:
                    sentenceid_to_pred_and_true[Y[i, 0]]["true"].add(tuple(Y[i, 1:]))
                
                if Y[i, 0] not in batch_dataset_sentenceids:
                    batch_sentenceid_to_dataset_sentenceid[len(unique_index)] = Y[i, 0]
                    unique_index.append(i)
                    batch_dataset_sentenceids.add(Y[i, 0])
            
            Ypred, Spred = model(X_wordid[unique_index], X_mask[unique_index], debug=True)
            Ygoldpred = model(X_wordid, X_mask, Y, use_gold_holders_and_targets=True)

            for y in Ypred:
                i = batch_sentenceid_to_dataset_sentenceid[y[0]]
                sentenceid_to_pred_and_true[i]["pred"].add(tuple(y[1:]))
            
            for y in Spred:
                i = batch_sentenceid_to_dataset_sentenceid[y[0]]
                sentenceid_to_pred_and_true[i]["span-score"].add(tuple(y[1:]))

            for y in Ygoldpred:
                sentenceid_to_pred_and_true[y[0]]["gold-pred"].add(tuple(y[1:]))
            
    return sentenceid_to_pred_and_true

# def write_predictions(dataset, sentenceid_to_pred_and_true, write_file, gold_key="true", pred_key="pred"):
#     '''
#     For each sentence, write the gold frames, pred frames, pred (with gold spans) frames, and span scores
#     '''
#     data = 

#     with open(write_file, "w") as fw:
#         for sentenceid, pred_and_true in sentenceid_to_pred_and_true.items():


def train(train_dataset, dev_dataset, test_dataset):
    '''
    Train role labeler model using train dataset
    Stop training when performance on dev dataset does not improve for `patience` epochs
    The best model is evaluated on both dev and test set

    The best model's weights is saved
    The dev and test set predictions are saved
    The dev and test set performance is saved
    The sentenceid to mpqa doc and sentence id mapping of dev and test set is saved

    The above is saved to config.RESULTS_FOLDER in a new folder whose name is the current timestamp
    '''

    model = RoleLabeler(config)
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataset))

    for epoch in range(config.max_n_epochs):

        print("epoch {:2d}".format(epoch + 1))

        model.train()
        train_loss = 0
        print_batches = len(train_dataset)//10

        for b, (X_wordid, X_mask, Y) in enumerate(train_dataset.data):

            model.zero_grad()
            loss = model(X_wordid, X_mask, Y, train=True)
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if (b  + 1) % print_batches == 0:
                print("\t batch {:3d} : train-loss = {:.4f}".format(b + 1, train_loss/(b + 1)))
        
        print("train-loss = {:.4f}".format(train_loss/len(train_dataset)))
    
        dev_sentenceid_to_pred_and_true = inference(dev_dataset, model)
        dev_df = evaluate(dev_sentenceid_to_pred_and_true)
        golddev_df = evaluate(dev_sentenceid_to_pred_and_true, pred_key="gold-pred")
        
        print("dev-eval:")
        print(dev_df)
        print()
        print("dev-eval (when given gold holder and target spans):")
        print(golddev_df)
        print()

    