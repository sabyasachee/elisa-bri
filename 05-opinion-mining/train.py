import math
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from models.OpinionMinerV2 import OpinionMiner
import config

def train(test_fold, dev_fold, fold_token_indices, fold_masks, fold_opinion_label_indices, fold_holder_label_indices, fold_entity_label_indices, fold_event_label_indices, fold_label_tuples, special_token_to_indices):
    n_folds = len(fold_token_indices)
    batch_size = config.batch_size
    train_folds = np.array([i for i in range(n_folds) if i not in [test_fold, dev_fold]])
    total_steps = sum(math.ceil(fold_token_indices[i].shape[0]/batch_size) for i in train_folds) * config.max_n_epochs

    model = OpinionMiner(config.pretrained_model_name, config.max_holder_span_size, config.max_target_span_size, config.max_n_holders_per_sentence, config.max_n_targets_per_sentence)
    model.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(config.max_n_epochs):
        print("epoch {}".format(epoch + 1))
        model.train()
        train_loss = 0
        np.random.shuffle(train_folds)

        for fold in train_folds:
            n_batches = math.ceil(fold_token_indices[fold].shape[0]/batch_size)
            print_n_batches = n_batches//5

            for b, batch_index in enumerate(np.random.permutation(n_batches)):
                token_indices = fold_token_indices[fold][batch_index * batch_size: (batch_index + 1) * batch_size]
                mask = fold_masks[fold][batch_index * batch_size: (batch_index + 1) * batch_size]
                label_tuples = fold_label_tuples[fold][batch_index * batch_size: (batch_index + 1) * batch_size]

                model.zero_grad()
                loss, pred_tuples = model(token_indices, mask, label_tuples)
                loss.backward()
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()
                scheduler.step()

                

        avg_train_loss = train_loss/total_steps
        print("train loss = {:.4f}".format(avg_train_loss))

        # model.eval()
        # print("dev set:")
        # dev_F1 = evaluate(dev_fold, model, fold_token_indices, fold_masks, fold_opinion_label_indices, fold_holder_label_indices, fold_entity_label_indices, fold_event_label_indices, fold_label_tuples)[-1,-1,-1]