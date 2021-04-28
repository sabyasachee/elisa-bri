import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

from .opinionBERT import opinionBERT

opinion_tagset = ['O', 'B-N', 'I-N', 'B-P', 'I-P', 'PAD']
bert_name = 'bert-base-cased'
num_labels = 6
num_layers = 3
hidden_size = 300
dropout_prob = 0.5
maxlen = 80
batch_size = 32
device = torch.device('cpu')

model = opinionBERT(bert_name, num_labels, num_layers, hidden_size, dropout_prob)
tokenizer = BertTokenizer.from_pretrained('/data/tokenizer/')
model.load_state_dict(torch.load('/data/opinion_expression.pt', map_location=device))
model.eval()

def find_opinion(bert_tokens_list):
    input_ids_arr = pad_sequences([tokenizer.convert_tokens_to_ids(tokens) for tokens in bert_tokens_list], maxlen=maxlen, dtype='long', value=0, truncating='post', padding='post')
    attn_mask_arr = np.array([[float(input_id != 0.0) for input_id in input_ids] for input_ids in input_ids_arr])

    input_ids_tensor = torch.LongTensor(input_ids_arr)
    attn_mask_tensor = torch.Tensor(attn_mask_arr)
    dataloader = DataLoader(TensorDataset(input_ids_tensor, attn_mask_tensor), batch_size=batch_size)
    prediction_ids_list = []

    for batch in tqdm(dataloader, desc='opinion'):
        batch_input_ids, batch_attn_mask = batch
        with torch.no_grad():
            logits = model(batch_input_ids, batch_attn_mask)
        logits = logits.detach().cpu().numpy()
        prediction_ids_list.extend([list(sentence_pred) for sentence_pred in np.argmax(logits, axis=2)])
    
    predictions_list = []
    for i, prediction_ids in enumerate(prediction_ids_list):
        predictions = [opinion_tagset[prediction_id] for prediction_id in prediction_ids]
        if len(predictions) < len(bert_tokens_list[i]):
            predictions += ['O'] * (len(bert_tokens_list[i]) - len(predictions))
        else:
            predictions = predictions[:len(bert_tokens_list[i])]
        predictions_list.append(predictions)

    new_predictions_list = []
    for bert_tokens, predictions in zip(bert_tokens_list, predictions_list):
        new_predictions = []
        for bert_token, prediction in zip(bert_tokens, predictions):
            if not bert_token.startswith('##'):
                new_predictions.append(prediction)
        new_predictions_list.append(new_predictions)

    return new_predictions_list

def find_opinion_tokenized(tokens_list):
    bert_tokens_list = [[bert_token for token in tokens for bert_token in tokenizer.tokenize(token)] for tokens in tokens_list]
    return find_opinion(bert_tokens_list)

def find_opinion_sentences(sentences):
    bert_tokens_list = [tokenizer.tokenize(sentence) for sentence in sentences]
    return find_opinion(bert_tokens_list)

def bert_tokenize(sentence):
    bert_tokens = tokenizer.tokenize(sentence)
    tokens = []
    for bert_token in bert_tokens:
        if bert_token.startswith('##'):
            tokens[-1] += bert_token[2:]
        else:
            tokens.append(bert_token)
    return tokens

def test():
    sentences = [
        ['I hate the man.'],
        ['The US fears the Chinese politics.'],
        ['I love the new dress!']
    ]

    print(find_opinion_tokenized(sentences))

if __name__ == '__main__':
    test()