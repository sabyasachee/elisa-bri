import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from .opinionLSTM import SRL4ORL_deep_tagger

tf.logging.set_verbosity(tf.logging.ERROR)

vocabulary = json.load(open('/data/vocabulary.json'))
embeddings = np.load('/data/embeddings.npy')
orl_label_set = ['O','B_DS','I_DS','B_H','I_H','B_T','I_T']
task_id = 1
window_size = 2
batch_size = 32

def pad_orl_data(batch, vocab):
    pad_id = vocab['<PAD>']

    max_length = max([len(inst[0]) for inst in batch])
    max_length_ds = max([len(inst[1]) for inst in batch])
    max_length_ctx = max([len(inst[3]) for inst in batch])

    batch_pad = []
    
    for (x, ds, ds_len, ctx, ctx_len, m) in batch:
        diff = max_length - len(x)
        z = []
        p = []
        for _ in range(diff):
            z.append(pad_id)
            p.append(0)

        diff_ds = max_length_ds - len(ds)
        q = []
        for _ in range(diff_ds):
            q.append(pad_id)

        diff_ctx = max_length_ctx - len(ctx)
        r = []
        for _ in range(diff_ctx):
            r.append(pad_id)

        batch_pad.append((  np.concatenate([x, z], 0), 
                            np.concatenate([ds, q], 0), 
                            ds_len, 
                            np.concatenate([ctx, r], 0), 
                            ctx_len, 
                            np.concatenate([m, p], 0), 
                            len(x)))

    return batch_pad

def find_orl(tokens_list, opinion_expression_indices_list):
    token_ids_list = []
    ds_token_ids_list = []
    ds_len_list = []
    context_ids_list = []
    context_len_list = []
    m_list = []

    for tokens, (start, end) in zip(tokens_list, opinion_expression_indices_list):
        token_ids = []
        ds_token_ids = []
        context_ids = []
        m = []

        for token in tokens:
            if token.lower() in vocabulary:
                token_ids.append(vocabulary[token.lower()])
            else:
                token_ids.append(vocabulary['<UNK>'])

        for i in range(start, end + 1):
            ds_token_ids.append(token_ids[i])

        for i in range(start - window_size, end + window_size + 1):
            if i < 0 or i > len(token_ids) - 1:
                context_ids.append(vocabulary['<PAD>'])
            else:
                context_ids.append(token_ids[i])

        for i in range(len(token_ids)):
            if start - window_size - 1 < i and i < end + window_size + 1:
                m.append(1)
            else:
                m.append(0)

        token_ids_list.append(token_ids)
        ds_token_ids_list.append(ds_token_ids)
        ds_len_list.append(len(ds_token_ids))
        context_ids_list.append(context_ids)
        context_len_list.append(len(context_ids))
        m_list.append(m)

    data = list(zip(token_ids_list, ds_token_ids_list, ds_len_list, context_ids_list, context_len_list, m_list))

    if len(data) % batch_size == 0:
        num_batches = int(len(data) / batch_size)
    else:
        num_batches = int(len(data) / batch_size) + 1

    batches = []

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(data))
        batch = data[start_index:end_index]
        batch_padded = pad_orl_data(batch, vocabulary)
        batches.append(batch_padded)
    
    predictions = []
    
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(24)

        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC',
                                    per_process_gpu_memory_fraction=0.3)

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=True,
                                    gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            seq_model = SRL4ORL_deep_tagger(106,
                                8,
                                embeddings,
                                False,
                                100,
                                'lstm',
                                24,
                                3,
                                0,
                                0.0,
                                0.0)
            init_vars = tf.global_variables_initializer()
            sess.run(init_vars)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, '/data/checkpoints/model')
            
            for batch in tqdm(batches, desc='orl'):
                sentences, ds, ds_len, ctx, ctx_len, m, sentence_lens = zip(*batch)

                feed_dict = {
                    seq_model.sentences: list(sentences),  # batch_data_padded_x,
                    # seq_model.labels: list(labels),  # batch_data_padded_y,
                    seq_model.sentence_lens: list(sentence_lens),  # batch_data_seqlens
                    seq_model.ds: list(ds),
                    seq_model.ds_len: list(ds_len),
                    seq_model.ctx: list(ctx),
                    seq_model.ctx_len: list(ctx_len),
                    seq_model.m: list(m),
                    seq_model.keep_rate_input: 1.0,
                    seq_model.keep_rate_output: 1.0,
                    seq_model.keep_state_rate: 1.0
                }
                transition_params_op = tf.get_default_graph().get_operation_by_name('task'+str(task_id)+'/transition_params').outputs[0]
                unary_scores_op = tf.get_default_graph().get_operation_by_name('task'+str(task_id)+'/unary_scores').outputs[0]

                tf_unary_scores, tf_transition_params = sess.run([unary_scores_op, transition_params_op], feed_dict)

                for i in range(len(sentences)):
                    length = sentence_lens[i]
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores[i, :length, :], tf_transition_params)
                    predictions.append(viterbi_sequence)

    predicted_labels_list = []

    for predicted_ids in predictions:
        predicted_labels = []
        for predicted_id in predicted_ids:
            predicted_labels.append(orl_label_set[predicted_id])
        predicted_labels_list.append(predicted_labels)

    return predicted_labels_list

def test():
    tokens_list = [
        ['I', 'hate', 'the', 'man', '.'],
        ['The', 'US', 'fears', 'the', 'Chinese', 'politics', '.'],
        ['I', 'love', 'the', 'new', 'dress', '!']
    ]

    opinion_expression_indices_list = [
        [1, 1],
        [2, 2],
        [1, 1]
    ]

    print(find_orl(tokens_list, opinion_expression_indices_list))