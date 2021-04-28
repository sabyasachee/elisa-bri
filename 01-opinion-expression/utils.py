import numpy as np

def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def is_overlap(x_span, y_span):
    a, b = x_span
    c, d = y_span
    return a <= d and c <= b

def opinion_metric(pred, true, tagset):
    tags = list(set([tag.split('-')[1] for tag in tagset if tag != 'O' and tag != 'PAD']))
    result = {}
    
    for tag in tags:
        n_pred_spans = 0
        n_true_spans = 0
        n_pred_spans_which_overlap_with_some_true_span = 0
        n_true_spans_which_overlap_with_some_pred_span = 0
        
        pred_span_length = 0
        true_span_length = 0
        overlap_length = 0
        
        for sentence_pred, sentence_true in zip(pred, true):
            pred_spans, true_spans = [], []
            
            i = 0
            while i < len(sentence_pred):
                if sentence_pred[i] == f'B-{tag}':
                    j = i + 1
                    while j < len(sentence_pred) and sentence_pred[j] == f'I-{tag}':
                        j += 1
                    pred_spans.append((i, j - 1))
                    i = j
                else:
                    i += 1
                    
            i = 0
            while i < len(sentence_true):
                if sentence_true[i] == f'B-{tag}':
                    j = i + 1
                    while j < len(sentence_true) and sentence_true[j] == f'I-{tag}':
                        j += 1
                    true_spans.append((i, j - 1))
                    i = j
                else:
                    i += 1
                    
            is_pred = np.zeros(len(sentence_pred), dtype=int)
            is_true = np.zeros(len(sentence_true), dtype=int)
            
            for pred_span in pred_spans:
                i, j = pred_span
                for k in range(i, j + 1):
                    is_pred[k] = 1
                    
            for true_span in true_spans:
                i, j = true_span
                for k in range(i, j + 1):
                    is_true[k] = 1
            
            n_pred_spans += len(pred_spans)
            n_true_spans += len(true_spans)
            n_pred_spans_which_overlap_with_some_true_span += sum(any(is_overlap(true_span, pred_span) for true_span in true_spans) for pred_span in pred_spans)
            n_true_spans_which_overlap_with_some_pred_span += sum(any(is_overlap(true_span, pred_span) for pred_span in pred_spans) for true_span in true_spans)
            
            pred_span_length += is_pred.sum()
            true_span_length += is_true.sum()
            overlap_length += (is_true & is_pred).sum()
            
        binary_precision = n_pred_spans_which_overlap_with_some_true_span/n_pred_spans
        binary_recall = n_true_spans_which_overlap_with_some_pred_span/n_true_spans
        binary_F1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        result[tag] = {'binary_precision': binary_precision, 'binary_recall': binary_recall, 'binary_F1': binary_F1}
        
        proportional_precision = overlap_length/pred_span_length
        proportional_recall = overlap_length/true_span_length
        proportional_F1 = 2 * proportional_precision * proportional_recall / (proportional_precision + proportional_recall)
        result[tag].update({'proportional_precision': proportional_precision, 'proportional_recall': proportional_recall, 'proportional_F1': proportional_F1})
    
    return result