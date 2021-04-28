import json
import re
from config import DATA_FOLDER, SRL4ORL_DATA_FOLDER

def preprocess_mpqa2(out_file, json_files):
    with open(out_file, 'w') as fw:
        fw.write(f'fileid\tsentence\ttoken\tlabel\n')
        
        for json_file in json_files:
            docs = json.load(open(json_file))
            
            for i in range(docs['documents_num']):
                doc = docs[f'document{i}']
                fileid = re.search('[^/]+/[^/]+$', doc['document_path']).group(0)
                
                for j in range(doc['sentences_num']):
                    sentence = doc[f'sentence{j}']
                    tokens = []
                    
                    for token in sentence['sentence_tokenized']:
                        if token == '-LRB-':
                            tokens.append('(')
                        elif token == '-RRB-':
                            tokens.append(')')
                        elif token == '-LSB-':
                            tokens.append('[')
                        elif token == '-RSB-':
                            tokens.append(']')
                        elif token == '-LCB-':
                            tokens.append('{')
                        elif token == '-RCB-':
                            tokens.append('}')
                        else:
                            tokens.append(token)
                    
                    expression_indices, polarities = [], []
                    
                    for k in sentence['dss_ids']:
                        dse = sentence[f'ds{k}']
                        ds_indices = dse['ds_indices']
                        ds_contains_pos_sentiment, ds_contains_neg_sentiment = False, False
                        
                        for l in range(dse['att_num']):
                            att = dse[f'att{l}']
                            ds_contains_pos_sentiment |= att['attitudes_types'] and att['attitudes_types'] == 'sentiment-pos'
                            ds_contains_neg_sentiment |= att['attitudes_types'] and att['attitudes_types'] == 'sentiment-neg'
                        if ds_contains_pos_sentiment != ds_contains_neg_sentiment:
                            expression_indices.append(ds_indices)
                            polarities.append('P' if ds_contains_pos_sentiment else 'N')
                    
                    overlapping_expression_ids_list = []
                    
                    for p in range(len(expression_indices)):
                        for q in range(len(overlapping_expression_ids_list)):
                            cluster_indices = set()
                            for r in overlapping_expression_ids_list[q]:
                                cluster_indices.update(expression_indices[r])
                            if cluster_indices.intersection(expression_indices[p]):
                                overlapping_expression_ids_list[q].append(p)
                                break
                        else:
                            overlapping_expression_ids_list.append([p])
                    
                    non_overlapping_expression_ids = []
                    
                    for expression_ids in overlapping_expression_ids_list:
                        expression_id = sorted(expression_ids, key=lambda p: (expression_indices[p][-1] - expression_indices[p][0] + 1, polarities[p] == 'N'))[0]
                        non_overlapping_expression_ids.append(expression_id)
                    
                    labels = ['O']*len(tokens)
                    
                    for p in non_overlapping_expression_ids:
                        polarity = polarities[p]
                        for q in expression_indices[p]:
                            labels[q] = f'I-{polarity}'
                        labels[expression_indices[p][0]] = f'B-{polarity}'
                    
                    for token, label in zip(tokens, labels):
                        fw.write(f'{fileid}\t{j}\t{token}\t{label}\n')

out_file = DATA_FOLDER + '/data.csv'
json_files = [SRL4ORL_DATA_FOLDER + '/new/dev.json', SRL4ORL_DATA_FOLDER + '/new/test_fold_0.json', SRL4ORL_DATA_FOLDER + '/new/train_fold_0.json']
preprocess_mpqa2(out_file, json_files)