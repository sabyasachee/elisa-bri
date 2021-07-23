MPQA3_FOLDER="/projects/opinion/data/30-mpqa/database.mpqa.3.0"
MPQA2_FOLDER="/projects/opinion/data/30-mpqa/database.mpqa.2.0"
MPQA3_PROCESSED_FOLDER="/projects/opinion/results/preprocessing-mpqa/mpqa3-processed"
MPQA2_PROCESSED_FOLDER="/projects/opinion/results/preprocessing-mpqa/mpqa2-processed"
MPQA_PROCESSED_FOLDER="/projects/opinion/results/preprocessing-mpqa"
RESULTS_FOLDER="/projects/opinion/results/opinion-mining"

max_sentence_length = 88
max_holder_span_size=10
max_target_span_size=20
max_n_holders_per_sentence=4
max_n_targets_per_sentence=10

pretrained_model_name="bert-base-cased"
device="cuda:0"
batch_size=128
lr=1e-5
patience=5
dropout=0.5
weight_decay=1e-2
max_n_epochs=20
max_grad_norm=1