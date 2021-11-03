import os
import config
import pandas as pd
from run import train
from data import RoleLabelerDataset

def entry():
    mpqa2_df = pd.read_csv(os.path.join(config.RESULTS_FOLDER, "folds/mpqa2.5fold.csv"), index_col=None)

    dev_docids = mpqa2_df.loc[mpqa2_df["fold"] == "dev", "docid"].values
    train_docids = mpqa2_df.loc[mpqa2_df["fold"] != "dev", "docid"].values
    test_docids = dev_docids

    dev_dataset = RoleLabelerDataset(dev_docids, mpqa2=True)
    train_dataset = RoleLabelerDataset(train_docids, mpqa2=True, ignore_negatives=True)

    train(train_dataset, dev_dataset, dev_dataset)

entry()