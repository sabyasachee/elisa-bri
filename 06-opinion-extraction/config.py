import os
import torch

# paths

PROJECT_FOLDER = "/proj/sbaruah/opinion"
RAW_FOLDER = os.path.join(PROJECT_FOLDER, "data/30-mpqa")
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "results/preprocessing-mpqa")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results/06-opinion-extraction-results")
SRL4ORL_SPLITS_FOLDER = os.path.join(PROJECT_FOLDER, "baselines/02-naacl-mpqa-srl4orl/datasplit/new")

# hparams

max_seq_len = 100
device = torch.device("cuda:0")

n_frame_arguments = 3
n_labels = 3
embed_model_name = "bert-base-cased"
encoder_hidden_size = 500
output_embedding_size = 10
encoder_num_layers = 2
decoder_hidden_size = 500
dropout = 0
beam_width = 5

lr = 1e-5
weight_decay = 1e-2
patience = 5
max_epochs = 20
max_grad_norm = 1
batch_size = 32