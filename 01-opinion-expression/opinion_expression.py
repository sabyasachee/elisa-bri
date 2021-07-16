import argparse
import torch
from transformer import train_and_evaluate_all_folds, evaluate_all_folds

def main():
    argparser = argparse.ArgumentParser(description='parameters for opinion expression model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    argparser.add_argument('--split_type', choices=['new', 'prior'], type=str, default='new')
    argparser.add_argument('--bert_name', default='bert-base-cased', type=str)
    argparser.add_argument('--batch_size', default=32, type=int)
    argparser.add_argument('--maxlen', default=80, type=int)
    argparser.add_argument('--num_layers', default=3, type=int)
    argparser.add_argument('--hidden_size', default=300, type=int)
    argparser.add_argument('--dropout_prob', default=0.5, type=float)
    argparser.add_argument('--device', default=-1, type=int)
    argparser.add_argument('--lr', default=1e-5, type=float)
    argparser.add_argument('--weight_decay', default=0.01, type=float)
    argparser.add_argument('--max_epochs', default=20, type=int)
    argparser.add_argument('--patience', default=5, type=int)
    argparser.add_argument('--use_class_weights', action='store_true', default=False)
    argparser.add_argument('--max_grad_norm', default=1., type=float)
    argparser.add_argument("--rnn_type", choices=["lstm", "gru"], default="gru")
    argparser.add_argument("--use_crf", action="store_true", default=False)
    argparser.add_argument("--bidirectional", action="store_true", default=False)
    argparser.add_argument("--freeze_bert", action="store_true", default=False)
    
    argparser.add_argument("--eval", action="store_true", default=False)
    argparser.add_argument("--model_name", type=str, default="model_09-04-2021_17:54:56")

    config = argparser.parse_args()
    config.device = torch.device(f'cuda:{config.device}') if config.device >= 0 else torch.device('cpu')
    print(config)

    if config.eval:
        evaluate_all_folds(config)
    else:
        train_and_evaluate_all_folds(config)

if __name__ == '__main__':
    main()