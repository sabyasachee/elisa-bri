import math
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import config
from data import OpinionDataset
from models.OpinionMinerV2 import OpinionMiner
from inference import inference
from evaluate import OpinionPerformance

def train(train_dataset: OpinionDataset, dev_dataset: OpinionDataset, test_dataset: OpinionDataset):
    batch_size = config.batch_size
    n_batches = math.ceil(len(train_dataset)/batch_size)
    total_steps = n_batches * config.max_n_epochs

    model = OpinionMiner()
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    pred_tuples = [None for _ in range(len(train_dataset))]

    for epoch in range(config.max_n_epochs):
        print("epoch {}, {} batches".format(epoch + 1, n_batches))
        model.train()
        train_loss = 0
        print_n_batches = n_batches//20

        for b, batch_index in enumerate(np.random.permutation(n_batches)):
            token_indices = train_dataset.token_indices[batch_index * batch_size: (batch_index + 1) * batch_size]
            mask = train_dataset.mask[batch_index * batch_size: (batch_index + 1) * batch_size]
            label_tuples = train_dataset.label_tuples[batch_index * batch_size: (batch_index + 1) * batch_size]

            model.zero_grad()
            loss, batch_pred_tuples = model(token_indices, mask, label_tuples)
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            pred_tuples[batch_index * batch_size: (batch_index + 1) * batch_size] = batch_pred_tuples

            if (b + 1) % print_n_batches == 0:
                print("\t\tbatch {:2d} loss = {:.3f}".format(b + 1, loss.item()))

        avg_train_loss = train_loss/n_batches
        print("train loss = {:.4f}".format(avg_train_loss))
    
        train_performance = OpinionPerformance(train_dataset.label_tuples, pred_tuples)
        print("train performance:")
        print(train_performance)

        dev_tuples = inference(model, dev_dataset)
        dev_performace = OpinionPerformance(dev_dataset.label_tuples, dev_tuples)
        print("dev performance:")
        print(dev_performace)

    test_tuples = inference(model, test_dataset)
    test_performace = OpinionPerformance(test_dataset.label_tuples, test_tuples)
    print("test performance:")
    print(test_performace)