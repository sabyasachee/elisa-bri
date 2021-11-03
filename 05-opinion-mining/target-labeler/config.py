# folder locations

PROJECT_FOLDER = "/workspace/opinion"
RESULTS_FOLDER = PROJECT_FOLDER + "/results/opinion-mining"
MPQA3_FOLDER = PROJECT_FOLDER + "/results/preprocessing-mpqa/mpqa3-processed"

# model hyperparameters

pretrained_model_name = "bert-base-cased"
max_seq_len = 100
position_embedding_size = 50
hidden_size = 500
bilistm_n_layers = 1

# training hyperparameters

device="cuda:0"
batch_size=128
lr=1e-5
patience=5
max_n_epochs=20

# regularization hyperparameters

dropout=0.5
max_grad_norm=1
weight_decay=1e-2