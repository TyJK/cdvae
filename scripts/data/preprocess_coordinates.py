""" Converts Coordinates and splits to train/val/test """

import pandas as pd
import argparse
import os

from cdvae.common.constants import ATOMIC_SYMBOL_TO_NUMBER_MAP

def main(args):
    print("reading csv...")
    data = pd.read_csv(args.data_path, low_memory=False)

    print("processing cols...")
    data["elements"] = data["elements"].apply(
        lambda elem_list: list(map(lambda sym: ATOMIC_SYMBOL_TO_NUMBER_MAP[sym], eval(elem_list)
    )))

    data["num_atoms"] = data["elements"].apply(len)
    del data["n_atoms"]

    data = data[data["num_atoms"] <= 80]  # trim largest molecules out

    data = data.round(8)  # smooths out compute tails

    data = data.rename(columns={"data_id": "dataset_id"})

    # stratified random sample 60/20/20
    print("splitting train/val/test...")
    train = data.sample(frac=.6)
    not_train_idx = list(set(data.index).difference(train.index))

    val = data.loc[not_train_idx].sample(frac=.5)
    test_idx = list(set(data.index).difference(train.index).difference(val.index))

    test = data.loc[test_idx]

    print("writing to disk...")
    train.to_csv(os.path.join(args.write_dir,"train.csv"), index=False)
    val.to_csv(os.path.join(args.write_dir,"val.csv"), index=False)
    test.to_csv(os.path.join(args.write_dir,"test.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default="data/dataset/")
    parser.add_argument('--write-dir', default="data/dataset/")
    args = parser.parse_args()

    main(args)
