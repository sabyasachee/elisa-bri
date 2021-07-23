import torch
import math

import config
from data import OpinionDataset

def inference(model, dataset: OpinionDataset):
    model.eval()
    batch_size = config.batch_size
    n_batches = math.ceil(len(dataset)/batch_size)
    pred_tuples = []

    for i in range(n_batches):
        token_indices = dataset.token_indices[i * batch_size: (i + 1) * batch_size]
        mask = dataset.mask[i * batch_size: (i + 1) * batch_size]

        with torch.no_grad():
            batch_pred_tuples = model(token_indices, mask)
            pred_tuples.extend(batch_pred_tuples)
    
    return pred_tuples