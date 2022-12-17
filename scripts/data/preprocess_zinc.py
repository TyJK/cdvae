""" Converts ZINC and splits to train/val/test """

import pandas as pd

from cdvae.common.constants import ATOMIC_SYMBOL_TO_NUMBER_MAP

def main():
    print("reading csv...")
    zinc = pd.read_csv("~/Projects/mila/molecule-representation-tda/data/raw/zinc-dec-13.csv", low_memory=False)

    print("processing cols...")
    zinc["elements"] = zinc["elements"].apply(
        lambda elem_list: list(map(lambda sym: ATOMIC_SYMBOL_TO_NUMBER_MAP[sym], eval(elem_list)
    )))

    zinc["num_atoms"] = zinc["elements"].apply(len)
    del zinc["n_atoms"]

    zinc = zinc[zinc["num_atoms"] <= 80]  # trim largest molecules out

    zinc = zinc.round(8)  # smooths out compute tails

    zinc = zinc.rename(columns={"zinc_id": "dataset_id"})

    # stratified random sample 60/20/20
    print("splitting train/val/test...")
    train = zinc.sample(frac=.6)
    not_train_idx = list(set(zinc.index).difference(train.index))

    val = zinc.loc[not_train_idx].sample(frac=.5)
    test_idx = list(set(zinc.index).difference(train.index).difference(val.index))

    test = zinc.loc[test_idx]

    print("writing to disk...")
    train.to_csv("~/Projects/mila/cdvae/data/zinc/train.csv", index=False)
    val.to_csv("~/Projects/mila/cdvae/data/zinc/val.csv", index=False)
    test.to_csv("~/Projects/mila/cdvae/data/zinc/test.csv", index=False)


if __name__ == "__main__":
    main()
