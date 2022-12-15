""" Converts MP and splits to train/val/test """

import pandas as pd

from cdvae.common.constants import ATOMIC_SYMBOL_TO_NUMBER_MAP

def main():
    print("reading csv...")
    mp = pd.read_csv("~/Projects/mila/molecule-representation-tda/data/raw/mp-dec-14.csv", low_memory=False)

    print("processing cols...")
    mp["elements"] = mp["elements"].apply(
        lambda elem_list: list(map(lambda sym: ATOMIC_SYMBOL_TO_NUMBER_MAP[sym], eval(elem_list)
    )))

    mp["num_atoms"] = mp["elements"].apply(len)

    mp = mp.round(8)  # smooths out compute tails

    mp = mp[mp["num_atoms"].between(12,100)]

    # stratified random sample 60/20/20
    print("splitting train/val/test...")
    train = mp.sample(frac=.6)
    not_train_idx = list(set(mp.index).difference(train.index))

    val = mp.loc[not_train_idx].sample(frac=.5)
    test_idx = list(set(mp.index).difference(train.index).difference(val.index))

    test = mp.loc[test_idx]

    print("writing to disk...")
    train.to_csv("~/Projects/mila/cdvae/data/mp/train.csv", index=False)
    val.to_csv("~/Projects/mila/cdvae/data/mp/val.csv", index=False)
    test.to_csv("~/Projects/mila/cdvae/data/mp/test.csv", index=False)


if __name__ == "__main__":
    main()
