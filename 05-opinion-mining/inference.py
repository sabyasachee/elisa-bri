import torch
import config
import math

def inference(model, fold, fold_token_indices, fold_masks):
    model.eval()
    batch_size = config.batch_size
    n_batches = math.ceil(fold_token_indices[fold].shape[0]/batch_size)
    pred_tuples = []

    for i in range(n_batches):
        token_indices = fold_token_indices[fold][i * batch_size: (i + 1) * batch_size]
        mask = fold_masks[fold][i * batch_size: (i + 1) * batch_size]

        with torch.no_grad():
            batch_pred_tuples = model(token_indices, mask)
            pred_tuples.extend(batch_pred_tuples)
    
    return pred_tuples