import opinion_expression.find_opinion_expressions
import opinion_roles.find_opinion_roles
from pprint import pprint

def find_opinions(sentences):
    if isinstance(sentences, str):
        sentences = [sentences]

    tokens_list = []
    opinions_list = []

    for sentence in sentences:
        tokens = opinion_expression.find_opinion_expressions.bert_tokenize(sentence)
        tokens_list.append(tokens)
        opinions_list.append({'sentence': sentence, 'tokenized': tokens, 'opinion_frames': []})

    opinion_predictions_list = opinion_expression.find_opinion_expressions.find_opinion_tokenized(tokens_list)
    opinion_sentence_ids, opinion_tokens_list, opinion_expression_indices_list, opinion_polarities = [], [], [], []

    for sentence_id, (tokens, predictions) in enumerate(zip(tokens_list, opinion_predictions_list)):
        i = 0
        while i < len(tokens):
            if predictions[i].startswith('B'):
                polarity = predictions[i].split('-')[1]
                j = i + 1
                while j < len(tokens) and predictions[j] == f'I-{polarity}':
                    j += 1
                opinion_sentence_ids.append(sentence_id)
                opinion_tokens_list.append(tokens)
                opinion_expression_indices_list.append([i, j - 1])
                opinion_polarities.append('positive' if polarity == 'P' else 'negative')
                i = j
            else:
                i += 1

    opinion_role_predictions_list = opinion_roles.find_opinion_roles.find_orl(opinion_tokens_list, opinion_expression_indices_list)
    opinion_holder_indices_list, opinion_target_indices_list = [], []

    for predictions in opinion_role_predictions_list:
        holder_indices, target_indices = [], []
        i = 0
        while i < len(predictions):
            if predictions[i] in ['B_H', 'B_T']:
                role = predictions[i].split('_')[1]
                j = i + 1
                while j < len(predictions) and predictions[j] == f'I_{role}':
                    j += 1
                if role == 'H':
                    holder_indices.append([i, j - 1])
                else:
                    target_indices.append([i, j - 1])
                i = j
            else:
                i += 1
        opinion_holder_indices_list.append(holder_indices)
        opinion_target_indices_list.append(target_indices)

    for sentence_id, tokens, opinion_expression_indices, polarity, holder_indices, target_indices in zip(opinion_sentence_ids, opinion_tokens_list, opinion_expression_indices_list, opinion_polarities, opinion_holder_indices_list, opinion_target_indices_list):
        oi, oj = opinion_expression_indices
        opinion_expression_tokens = tokens[oi: oj + 1]
        holder_indices_list = [list(range(hi, hj + 1)) for hi, hj in holder_indices]
        holder_tokens_list = [tokens[hi: hj + 1] for hi, hj in holder_indices]
        target_indices_list = [list(range(ti, tj + 1)) for ti, tj in target_indices]
        target_tokens_list = [tokens[ti: tj + 1] for ti, tj in target_indices]
        holders = [{'indices': holder_indices, 'tokens': holder_tokens} for holder_indices, holder_tokens in zip(holder_indices_list, holder_tokens_list)]
        targets = [{'indices': target_indices, 'tokens': target_tokens} for target_indices, target_tokens in zip(target_indices_list, target_tokens_list)]
        opinion_frame = {'expression': {'indices': list(range(oi, oj + 1)), 'tokens': opinion_expression_tokens}, 'polarity': polarity, 'holders': holders, 'targets': targets}
        opinions_list[sentence_id]['opinion_frames'].append(opinion_frame)

    print(opinion_predictions_list)
    print(opinion_expression_indices_list)
    return opinions_list

if __name__ == '__main__':
    sentences = ['She hates him', 'The US fears Chinese politics']
    pprint(find_opinions(sentences))