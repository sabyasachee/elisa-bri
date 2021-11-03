import os
import config
import pandas as pd
from data import OpinionDataset
from train import train

def main():
    mpqa2_doc_ids = open(os.path.join(config.MPQA2_FOLDER, "doclist.attitudeSubset")).read().strip().split("\n")
    mpqa3_df = pd.read_csv(os.path.join(config.RESULTS_FOLDER, "mpqa3/2fold.csv"), index_col=None)
    train_doc_ids = [os.path.join("mpqa2-processed", doc_id) for doc_id in mpqa2_doc_ids if doc_id not in mpqa3_df["doc_id"].values]
    dev_doc_ids = [os.path.join("mpqa3-processed", row["doc_id"]) for _, row in mpqa3_df.iterrows() if row["fold"] == 0]
    test_doc_ids = [os.path.join("mpqa3-processed", row["doc_id"]) for _, row in mpqa3_df.iterrows() if row["fold"] == 1]

    train_dataset = OpinionDataset(train_doc_ids)
    dev_dataset = OpinionDataset(dev_doc_ids)
    test_dataset = OpinionDataset(test_doc_ids)

    # train(train_dataset, dev_dataset, test_dataset)

if __name__ == "__main__":
    main()