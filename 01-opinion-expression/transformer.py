import os
import pandas as pd
import numpy as np
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences

from opinionBERT import opinionBERT

from config import DATA_FOLDER, SRL4ORL_DATASPLIT_FOLDER, MODEL_FOLDER
from utils import tokenize_and_preserve_labels, opinion_metric

def prepare_data(data_file, datasplit_dir, bert_name, maxlen, batch_size):

    def get_dataloader(fileids, shuffle=False):
        idx = np.isin(fileids_arr, fileids)
        input_ids_tensor = torch.LongTensor(input_ids_arr[idx].reshape(-1, maxlen))
        attn_mask_tensor = torch.FloatTensor(attn_mask_arr[idx].reshape(-1, maxlen))
        labels_tensor = torch.LongTensor(labels_arr[idx].reshape(-1, maxlen))
        return DataLoader(TensorDataset(input_ids_tensor, attn_mask_tensor, labels_tensor), shuffle=shuffle, batch_size=batch_size)

    df = pd.read_csv(data_file, index_col=None, sep='\t')

    tagset = df['label'].unique().tolist()
    tagset.append('PAD')
    tag2idx = dict([(t, i) for i, t in enumerate(tagset)])

    dev_fileids = open(os.path.join(datasplit_dir, 'filelist_dev')).read().strip().split('\n')
    n_folds = sum(['train' in file for file in os.listdir(datasplit_dir)])
    train_fileids_list = [open(os.path.join(datasplit_dir, f'filelist_train{fold}')).read().strip().split('\n') for fold in range(n_folds)]
    test_fileids_list = [open(os.path.join(datasplit_dir, f'filelist_test{fold}')).read().strip().split('\n') for fold in range(n_folds)]

    fileids, tokens_list, labels_list = [], [], []

    tokenizer = BertTokenizer.from_pretrained(bert_name)

    for (fileid, _), sentence_df in df.groupby(['fileid','sentence']):
        tokens = sentence_df.token.tolist()
        labels = sentence_df.label.tolist()
        tokens, labels = tokenize_and_preserve_labels(tokenizer, tokens, labels)
        tokens_list.append(tokens)
        labels_list.append(labels)
        fileids.append(fileid)

    input_ids_arr = pad_sequences([tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list], maxlen=maxlen, dtype='long', value=0, truncating='post', padding='post')
    labels_arr = pad_sequences([[tag2idx[label] for label in labels] for labels in labels_list], maxlen=maxlen, dtype='long', value=tag2idx['PAD'], truncating='post', padding='post')
    attn_mask_arr = np.array([[float(input_id != 0.0) for input_id in input_ids] for input_ids in input_ids_arr])
    fileids_arr = np.tile(np.array(fileids).reshape(-1, 1), (1, input_ids_arr.shape[1]))

    dev_dataloader = get_dataloader(dev_fileids)
    train_dataloaders = [get_dataloader(train_fileids, shuffle=True) for train_fileids in train_fileids_list]
    test_dataloaders = [get_dataloader(test_fileids) for test_fileids in test_fileids_list]

    class_weights_list = []
    for train_dataloader in train_dataloaders:
        class_weights = np.zeros(len(tag2idx))
        for batch in train_dataloader:
            _, _, batch_labels = batch
            batch_labels = batch_labels.numpy().astype(int).flatten()
            for label in batch_labels:
                class_weights[label] += 1
                
        class_weights = np.sum(class_weights)/class_weights
        class_weights = torch.FloatTensor(class_weights)
        class_weights_list.append(class_weights)

    return dev_dataloader, train_dataloaders, test_dataloaders, class_weights_list, tagset

def plot_data(model_dir, train_set_avg_train_losses, train_set_avg_eval_losses, train_set_avg_binary_f1s, dev_set_avg_eval_losses, dev_set_avg_binary_f1s):
    epochs = np.arange(len(train_set_avg_train_losses)) + 1
    
    plt.close('all')
    plt.plot(epochs, train_set_avg_train_losses, 'b-o', label='avg train loss')
    plt.plot(epochs, train_set_avg_eval_losses, 'g-o', label='train set: avg eval loss')
    plt.plot(epochs, dev_set_avg_eval_losses, 'r-o', label='dev set: avg eval loss')
    plt.legend()
    plt.title('loss vs epoch')
    plt.savefig(os.path.join(model_dir, 'loss.png'))

    plt.close('all')
    plt.plot(epochs, train_set_avg_binary_f1s, 'b-o', label='train set: avg binary F1')
    plt.plot(epochs, dev_set_avg_binary_f1s, 'b-o', label='train set: avg binary F1')
    plt.legend()
    plt.title('F1 vs epoch')
    plt.savefig(os.path.join(model_dir, 'f1.png'))

def evaluate(dataloader, model, tagset, config):
    eval_loss = 0
    pred_list, labels_list = [], []

    for batch in dataloader:
        batch_input_ids, batch_attn_mask, batch_labels = [tensor.to(config.device) for tensor in batch]
        batch_crf_attn_mask = batch_attn_mask.byte().to(config.device)
        with torch.no_grad():
            loss, pred = model(batch_input_ids, batch_attn_mask, batch_crf_attn_mask, batch_labels)
        batch_labels = batch_labels.detach().cpu().numpy()
        pred_list.extend(pred)
        labels_list.extend([list(sentence_labels) for sentence_labels in batch_labels])
        eval_loss += loss.item()
    
    avg_eval_loss = eval_loss/len(dataloader)
    print(f'\tavg eval loss = {avg_eval_loss:.4f}')

    pred_list = [[tagset[int(p)] for p, l in zip(pred, labels) if tagset[int(l)] != "PAD"] for pred, labels in zip(pred_list, labels_list)]
    labels_list = [[tagset[int(l)] for l in labels if tagset[int(l)] != "PAD"] for labels in labels_list]

    eval_result = opinion_metric(pred_list, labels_list, tagset)
    binary_f1 = 0

    for label, label_eval_result in eval_result.items():
        print(f'\t{label} binary: precision = {label_eval_result["binary_precision"]:.4f}, recall = {label_eval_result["binary_recall"]:.4f}, F1 = {label_eval_result["binary_F1"]:.4f}')
        print(f'\t{label} proportional: precision = {label_eval_result["proportional_precision"]:.4f}, recall = {label_eval_result["proportional_recall"]:.4f}, F1 = {label_eval_result["proportional_F1"]:.4f}')
        binary_f1 += label_eval_result["binary_F1"]
    
    avg_binary_f1 = binary_f1/len(eval_result)
    return avg_binary_f1, avg_eval_loss, eval_result

def train_and_evaluate_one_fold(dev_dataloader, train_dataloader, test_dataloader, class_weights, tagset, model, optimizer, scheduler, model_dir, config):
    patience = config.patience
    best_avg_binary_F1 = -np.inf
    best_epoch = -1

    train_set_avg_train_losses, train_set_avg_eval_losses, train_set_avg_binary_f1s = [], [], []
    dev_set_avg_eval_losses, dev_set_avg_binary_f1s = [], []

    for epoch in trange(config.max_epochs, desc='Epoch'):
        model.train()
        train_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch_input_ids, batch_attn_mask, batch_labels = [tensor.to(config.device) for tensor in batch]
            batch_crf_attn_mask = batch_attn_mask.byte().to(config.device)
            model.zero_grad()
            if config.use_class_weights:
                loss, _ = model(batch_input_ids, batch_attn_mask, batch_crf_attn_mask, batch_labels, class_weights=class_weights.to(config.device))
            else:
                loss, _ = model(batch_input_ids, batch_attn_mask, batch_crf_attn_mask, batch_labels)
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss/len(train_dataloader)
        print(f'avg train loss = {avg_train_loss:.4f}')

        model.eval()
        print('evaluating train set')
        train_set_avg_binary_f1, train_set_avg_eval_loss, train_set_eval_result = evaluate(train_dataloader, model, tagset, config)
        print('evaluating dev set')
        dev_set_avg_binary_f1, dev_set_avg_eval_loss, dev_set_eval_result = evaluate(dev_dataloader, model, tagset, config)
        
        train_set_avg_train_losses.append(avg_train_loss)
        train_set_avg_eval_losses.append(train_set_avg_eval_loss)
        train_set_avg_binary_f1s.append(train_set_avg_binary_f1)
        dev_set_avg_eval_losses.append(dev_set_avg_eval_loss)
        dev_set_avg_binary_f1s.append(dev_set_avg_binary_f1)

        if best_avg_binary_F1 < dev_set_avg_binary_f1:
            best_avg_binary_F1 = dev_set_avg_binary_f1
            best_epoch = epoch
            print('dev set avg binary f1 improved')
            patience = config.patience

            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, 'weights.pt'))
            
            with open(os.path.join(model_dir, 'log.txt'), 'w') as fw:
                for config_key, config_value in vars(config).items():
                    fw.write(f'{config_key:30s}: {config_value}\n')
                fw.write('\n\n')
                
                fw.write(f'epoch = {best_epoch}\n\n')
                fw.write(f'train set:\n')
                for label, label_eval_result in train_set_eval_result.items():
                    fw.write(f'\t{label} binary: precision = {label_eval_result["binary_precision"]:.4f}, recall = {label_eval_result["binary_recall"]:.4f}, F1 = {label_eval_result["binary_F1"]:.4f}\n')
                    fw.write(f'\t{label} proportional: precision = {label_eval_result["proportional_precision"]:.4f}, recall = {label_eval_result["proportional_recall"]:.4f}, F1 = {label_eval_result["proportional_F1"]:.4f}\n')
                fw.write('\n\n')

                fw.write(f'dev set:\n')
                for label, label_eval_result in dev_set_eval_result.items():
                    fw.write(f'\t{label} binary: precision = {label_eval_result["binary_precision"]:.4f}, recall = {label_eval_result["binary_recall"]:.4f}, F1 = {label_eval_result["binary_F1"]:.4f}\n')
                    fw.write(f'\t{label} proportional: precision = {label_eval_result["proportional_precision"]:.4f}, recall = {label_eval_result["proportional_recall"]:.4f}, F1 = {label_eval_result["proportional_F1"]:.4f}\n')
                fw.write('\n\n')
        
        else:
            patience -= 1

        if patience == 0:
            print('early stop')
            break
    
    plot_data(model_dir, train_set_avg_train_losses, train_set_avg_eval_losses, train_set_avg_binary_f1s, dev_set_avg_eval_losses, dev_set_avg_binary_f1s)

    model.eval()
    print('evaluating test set')
    _, _, test_set_eval_result = evaluate(test_dataloader, model, tagset, config)
    with open(os.path.join(model_dir, 'log.txt'), 'a') as fw:
        fw.write(f'test set:\n')
        for label, label_eval_result in test_set_eval_result.items():
            fw.write(f'\t{label} binary: precision = {label_eval_result["binary_precision"]:.4f}, recall = {label_eval_result["binary_recall"]:.4f}, F1 = {label_eval_result["binary_F1"]:.4f}\n')
            fw.write(f'\t{label} proportional: precision = {label_eval_result["proportional_precision"]:.4f}, recall = {label_eval_result["proportional_recall"]:.4f}, F1 = {label_eval_result["proportional_F1"]:.4f}\n')

def train_and_evaluate_all_folds(config):
    model_dir = os.path.join(MODEL_FOLDER, datetime.now().strftime("model_%d-%m-%Y_%H:%M:%S"))

    data_file = os.path.join(DATA_FOLDER, 'data.csv')
    datasplit_dir = os.path.join(SRL4ORL_DATASPLIT_FOLDER, config.split_type)
    dev_dataloader, train_dataloaders, test_dataloaders, class_weights_list, tagset = prepare_data(data_file, datasplit_dir, config.bert_name, config.maxlen, config.batch_size)

    for fold, (train_dataloader, test_dataloader, class_weights) in enumerate(zip(train_dataloaders, test_dataloaders, class_weights_list)):
        print(f'fold {fold}')
        fold_model_dir = os.path.join(model_dir, f'fold{fold}')
        model = opinionBERT(bert_name = config.bert_name, num_labels = len(tagset), num_layers = config.num_layers, hidden_size = config.hidden_size, dropout_prob = config.dropout_prob, rnn_type = config.rnn_type, bidirectional = config.bidirectional, use_crf = config.use_crf, freeze_bert = config.freeze_bert)
        model.to(config.device)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        total_steps = len(train_dataloader) * config.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        train_and_evaluate_one_fold(dev_dataloader, train_dataloader, test_dataloader, class_weights, tagset, model, optimizer, scheduler, fold_model_dir, config)
        print(f'fold {fold} done\n\n\n\n')

def evaluate_all_folds(config):
    model_dir = os.path.join(MODEL_FOLDER, config.model_name)
    data_file = os.path.join(DATA_FOLDER, 'data.csv')
    datasplit_dir = os.path.join(SRL4ORL_DATASPLIT_FOLDER, config.split_type)
    dev_dataloader, train_dataloaders, test_dataloaders, class_weights_list, tagset = prepare_data(data_file, datasplit_dir, config.bert_name, config.maxlen, config.batch_size)

    for fold, (train_dataloader, test_dataloader, class_weights) in enumerate(zip(train_dataloaders, test_dataloaders, class_weights_list)):
        print(f'fold {fold}')
        fold_model_dir = os.path.join(model_dir, f'fold{fold}')
        model = opinionBERT(bert_name = config.bert_name, num_labels = len(tagset), num_layers = config.num_layers, hidden_size = config.hidden_size, dropout_prob = config.dropout_prob, rnn_type = config.rnn_type, bidirectional = config.bidirectional, use_crf = config.use_crf, freeze_bert = config.freeze_bert)
        model.load_state_dict(torch.load(os.path.join(fold_model_dir, "weights.pt")))
        _, _, test_set_eval_result = evaluate(test_dataloader, model, tagset, config)
        print(test_set_eval_result)