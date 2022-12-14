""" Converts QM9 and splits to train/val/test """

import pandas as pd

from cdvae.common.constants import ATOMIC_SYMBOL_TO_NUMBER_MAP

def main():
    print("reading csv...")
    qm9 = pd.read_csv("~/Projects/mila/molecule-representation-tda/data/raw/qm9-dec-11.csv", low_memory=False)

    print("processing cols...")
    qm9["elements"] = qm9["elements"].apply(
        lambda elem_list: list(map(lambda sym: ATOMIC_SYMBOL_TO_NUMBER_MAP[sym], eval(elem_list)
    )))

    qm9["num_atoms"] = qm9["elements"].apply(len)

    qm9 = qm9.round(8)  # smooths out compute tails

    # stratified random sample 60/20/20
    print("splitting train/val/test...")
    train = qm9.sample(frac=.6)
    not_train_idx = list(set(qm9.index).difference(train.index))

    val = qm9.loc[not_train_idx].sample(frac=.5)
    test_idx = list(set(qm9.index).difference(train.index).difference(val.index))

    test = qm9.loc[test_idx]

    print("writing to disk...")
    train.to_csv("~/Projects/mila/cdvae/data/qm9/train.csv", index=False)
    val.to_csv("~/Projects/mila/cdvae/data/qm9/val.csv", index=False)
    test.to_csv("~/Projects/mila/cdvae/data/qm9/test.csv", index=False)


if __name__ == "__main__":
    main()
