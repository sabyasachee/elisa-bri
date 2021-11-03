import torch

# folder

PROJECT_FOLDER = "/workspace/opinion"
DATA_FOLDER = PROJECT_FOLDER + "/data/30-mpqa"
PROCESSED_FOLDER = PROJECT_FOLDER + "/results/preprocessing-mpqa"
RESULTS_FOLDER = PROJECT_FOLDER + "/results/opinion-mining"

# model hyper params

pretrained_model_name = "SpanBERT/spanbert-large-cased"
scorer_hidden_size = 500
device = torch.device("cuda:0")

max_seq_len = 80
max_holder_length = 8
max_target_length = 17
n_holders = 2
n_targets = 2

# training hyper params

batch_size = 32
lr = 1e-5
weight_decay = 1e-2
max_n_epochs = 10
patience = 5
max_grad_norm = 1e-1