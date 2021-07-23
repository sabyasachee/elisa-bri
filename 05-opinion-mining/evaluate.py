class Metric:

    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall
        self.f1 = (2 * precision * recall) / (1e-23 + precision + recall)
    
    def __repr__(self) -> str:
        return "P={:5.1f}, R={:5.1f}, F1={:5.1f}".format(100*self.precision, 100*self.recall, 100*self.f1)

class Performance:

    def __init__(self, binary_precision, binary_recall, proportional_precision, proportional_recall, exact_precision, exact_recall):
        self.binary = Metric(binary_precision, binary_recall)
        self.proportional = Metric(proportional_precision, proportional_recall)
        self.exact = Metric(exact_precision, exact_recall)
    
    def __repr__(self) -> str:
        return "binary: {}  proportional: {}  exact: {}".format(self.binary, self.proportional, self.exact)
        
class OpinionPerformance:

    def __init__(self, label_tuples_list, pred_tuples_list):
        self.label_tuples_list = label_tuples_list
        self.pred_tuples_list = pred_tuples_list
        self.evaluate(label_tuples_list, pred_tuples_list)
    
    def evaluate(self, gold_opinion_tuples_list, pred_opinion_tuples_list):
        gold_targets, pred_targets = set(), set()
        gold_holders, pred_holders = set(), set()
        gold_implicit_spans, pred_implicit_spans, gold_implicit_labels, pred_implicit_labels = [], [], [], []
        gold_explicit_pairs, pred_explicit_pairs, gold_explicit_labels, pred_explicit_labels = [], [], [], []

        for i, (gold_opinion_tuples, pred_opinion_tuples) in enumerate(zip(gold_opinion_tuples_list, pred_opinion_tuples_list)):
            
            for span, label in gold_opinion_tuples["target_to_label"].items():
                if label > 0:
                    gold_targets.add((i, span[0], span[1]))
                    gold_implicit_spans.append((i, span[0], span[1]))
                    gold_implicit_labels.append(label)
        
            for span_pair, label in gold_opinion_tuples["holder_and_target_to_label"].items():
                holder_span = (i, span_pair[0], span_pair[1])
                target_span = (i, span_pair[2], span_pair[3])
                if label > 0:
                    gold_targets.add(target_span)
                    gold_holders.add(holder_span)
                    gold_explicit_pairs.append((holder_span, target_span))
                    gold_explicit_labels.append(label)

            for span, label in pred_opinion_tuples["target_to_label"].items():
                if label > 0:
                    pred_targets.add((i, span[0], span[1]))
                    pred_implicit_spans.append((i, span[0], span[1]))
                    pred_implicit_labels.append(label)
        
            for span_pair, label in pred_opinion_tuples["holder_and_target_to_label"].items():
                holder_span = (i, span_pair[0], span_pair[1])
                target_span = (i, span_pair[2], span_pair[3])
                if label > 0:
                    pred_targets.add(target_span)
                    pred_holders.add(holder_span)
                    pred_explicit_pairs.append((holder_span, target_span))
                    pred_explicit_labels.append(label)
        
        gold_targets = list(gold_targets)
        gold_holders = list(gold_holders)
        pred_targets = list(pred_targets)
        pred_holders = list(pred_holders)

        self.target_identification = self.evaluate_spans(gold_targets, pred_targets)
        self.holder_identification = self.evaluate_spans(gold_holders, pred_holders)
        self.implicit_identification = self.evaluate_spans(gold_implicit_spans, pred_implicit_spans)
        self.explicit_identification = self.evaluate_span_pairs(gold_explicit_pairs, pred_explicit_pairs)
        self.implicit_classification = self.evaluate_spans(gold_implicit_spans, pred_implicit_spans, gold_implicit_labels, pred_implicit_labels, labelset=[1,2,3], match_labels=True)
        self.explicit_classification = self.evaluate_span_pairs(gold_explicit_pairs, pred_explicit_pairs, gold_explicit_labels, pred_explicit_labels, labelset=[1,2,3], match_labels=True)
    
    def __repr__(self) -> str:
        line1 = "Target Identification   => {}".format(self.target_identification)
        line2 = "Holder Identification   => {}".format(self.holder_identification)
        line3 = "Implicit Identification => {}".format(self.implicit_identification)
        line4 = "Explicit Identification => {}".format(self.explicit_identification)
        line5 = "Implicit Classification => {}".format(self.implicit_classification)
        line6 = "Explicit Classification => {}".format(self.explicit_classification)
        return "\n".join([line1, line2, line3, line4, line5, line6])

    def is_overlap(self, span_x, span_y):
        return span_x[0] == span_y[0] and span_x[1] < span_y[2] and span_y[1] < span_x[2]

    def overlap_len(self, span_x, span_y):
        return (span_x[0] == span_y[0]) * len(set(range(span_x[1], span_x[2])).intersection(range(span_y[1], span_y[2])))

    def span_length(self, span):
        return span[2] - span[1]

    def evaluate_spans(self, gold_spans, pred_spans, gold_labels=[], pred_labels=[], labelset=[], match_labels=False):
        if match_labels and isinstance(labelset, list) and len(labelset):
            gold_indices = [i for i in range(len(gold_labels)) if gold_labels[i] in labelset]
            pred_indices = [i for i in range(len(pred_labels)) if pred_labels[i] in labelset]
        else:
            gold_indices = list(range(len(gold_spans)))
            pred_indices = list(range(len(pred_spans)))

        binary_gold_overlap, binary_pred_overlap = 0, 0
        proportional_gold_overlap, proportional_pred_overlap = 0, 0
        exact_gold_overlap, exact_pred_overlap = 0, 0

        for i in gold_indices:
            gold_span = gold_spans[i]
            binary_gold_score, proportional_gold_score, exact_gold_score = 0, 0, 0
            for j in pred_indices:
                pred_span = pred_spans[j]
                if self.is_overlap(gold_span, pred_span) and (not match_labels or gold_labels[i] == pred_labels[j]):
                    binary_gold_score = 1
                    proportional_gold_score = max(proportional_gold_score, self.overlap_len(gold_span, pred_span)/self.span_length(gold_span))
                    exact_gold_score = max(exact_gold_score, int(gold_span == pred_span))
            binary_gold_overlap += binary_gold_score
            proportional_gold_overlap += proportional_gold_score
            exact_gold_overlap += exact_gold_score
        
        for i in pred_indices:
            pred_span = pred_spans[i]
            binary_pred_score, proportional_pred_score, exact_pred_score = 0, 0, 0
            for j in gold_indices:
                gold_span = gold_spans[j]
                if self.is_overlap(pred_span, gold_span) and (not match_labels or pred_labels[i] == gold_labels[j]):
                    binary_pred_score = 1
                    proportional_pred_score = max(proportional_pred_score, self.overlap_len(pred_span, gold_span)/self.span_length(pred_span))
                    exact_pred_score = max(exact_pred_score, int(pred_span == gold_span))
            binary_pred_overlap += binary_pred_score
            proportional_pred_overlap += proportional_pred_score
            exact_pred_overlap += exact_pred_score
        
        binary_recall, binary_precision = binary_gold_overlap/(1e-23 + len(gold_spans)), binary_pred_overlap/(1e-23 + len(pred_spans))
        proportional_recall, proportional_precision = proportional_gold_overlap/(1e-23 + len(gold_spans)), proportional_pred_overlap/(1e-23 + len(pred_spans))
        exact_recall, exact_precision = exact_gold_overlap/(1e-23 + len(gold_spans)), exact_pred_overlap/(1e-23 + len(pred_spans))

        return Performance(binary_precision, binary_recall, proportional_precision, proportional_recall, exact_precision, exact_recall)

    def evaluate_span_pairs(self, gold_span_pairs, pred_span_pairs, gold_labels=[], pred_labels=[], labelset=[], match_labels=False):
        if match_labels and isinstance(labelset, list) and len(labelset):
            gold_indices = [i for i in range(len(gold_labels)) if gold_labels[i] in labelset]
            pred_indices = [i for i in range(len(pred_labels)) if pred_labels[i] in labelset]
        else:
            gold_indices = list(range(len(gold_span_pairs)))
            pred_indices = list(range(len(pred_span_pairs)))

        binary_gold_overlap, binary_pred_overlap = 0, 0
        proportional_gold_overlap, proportional_pred_overlap = 0, 0
        exact_gold_overlap, exact_pred_overlap = 0, 0

        for i in gold_indices:
            gold_span_x, gold_span_y = gold_span_pairs[i]
            binary_gold_score, proportional_gold_score, exact_gold_score = 0, 0, 0
            for j in pred_indices:
                pred_span_x, pred_span_y = pred_span_pairs[j]
                if self.is_overlap(gold_span_x, pred_span_x) and self.is_overlap(gold_span_y, pred_span_y) and (not match_labels or gold_labels[i] == pred_labels[j]):
                    binary_gold_score = 1
                    proportional_gold_score = max(proportional_gold_score, (self.overlap_len(gold_span_x, pred_span_x) + self.overlap_len(gold_span_y, pred_span_y))/(self.span_length(gold_span_x) + self.span_length(gold_span_y)))
                    exact_gold_score = max(exact_gold_score, int(gold_span_x == pred_span_x and gold_span_y == pred_span_y))
            binary_gold_overlap += binary_gold_score
            proportional_gold_overlap += proportional_gold_score
            exact_gold_overlap += exact_gold_score
        
        for i in pred_indices:
            pred_span_x, pred_span_y = pred_span_pairs[i]
            binary_pred_score, proportional_pred_score, exact_pred_score = 0, 0, 0
            for j in gold_indices:
                gold_span_x, gold_span_y = gold_span_pairs[j]
                if self.is_overlap(pred_span_x, gold_span_x) and self.is_overlap(pred_span_y, gold_span_y) and (not match_labels or pred_labels[i] == gold_labels[j]):
                    binary_pred_score = 1
                    proportional_pred_score = max(proportional_pred_score, (self.overlap_len(gold_span_x, pred_span_x) + self.overlap_len(gold_span_y, pred_span_y))/(self.span_length(pred_span_x) + self.span_length(pred_span_y)))
                    exact_pred_score = max(exact_pred_score, int(gold_span_x == pred_span_x and gold_span_y == pred_span_y))
            binary_pred_overlap += binary_pred_score
            proportional_pred_overlap += proportional_pred_score
            exact_pred_overlap += exact_pred_score

        binary_recall, binary_precision = binary_gold_overlap/(1e-23 + len(gold_span_pairs)), binary_pred_overlap/(1e-23 + len(pred_span_pairs))
        proportional_recall, proportional_precision = proportional_gold_overlap/(1e-23 + len(gold_span_pairs)), proportional_pred_overlap/(1e-23 + len(pred_span_pairs))
        exact_recall, exact_precision = exact_gold_overlap/(1e-23 + len(gold_span_pairs)), exact_pred_overlap/(1e-23 + len(pred_span_pairs))

        return Performance(binary_precision, binary_recall, proportional_precision, proportional_recall, exact_precision, exact_recall)