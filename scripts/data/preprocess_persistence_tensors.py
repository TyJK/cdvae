import argparse
import os
import pickle
from tqdm import tqdm

import numpy as np

def main(args):
    pi_files = os.listdir(args.data_dir)
    pi_files.sort()

    all_keys = []
    all_pis = []
    all_pis_full = []
    for pif in tqdm(pi_files):
        pif = open(os.path.join(args.data_dir,pif), "rb")
        pi_dict = pickle.load(pif)
        pif.close()

        keys = list(pi_dict.keys())
        keys.sort()

        pis = []
        pis_full = []
        for key in keys:
            key_pis = pi_dict[key]
            key_pis_plain = np.copy(key_pis[-1,:])  # last channel in BCWH tensor is plain PI
            pis.append(key_pis_plain)
            pis_full.append(key_pis)
            # del key_pis

        all_keys.extend(keys)
        all_pis.append(np.array(pis))
        all_pis_full.append(np.array(pis_full))

    all_keys = np.array(all_keys)
    all_pis = np.concatenate(all_pis).astype(float)
    all_pis_full = np.concatenate(all_pis_full).astype(float)

    if not os.path.exists(args.write_dir):
        os.makedirs(args.write_dir)

    with open(os.path.join(args.write_dir,"all_keys.npy"), "wb") as f:
        np.save(f, all_keys)

    with open(os.path.join(args.write_dir,"all_pis.npy"), "wb") as f:
        np.save(f, all_pis_full)

    with open(os.path.join(args.write_dir,"all_pis_plain.npy"), "wb") as f:
        np.save(f, all_pis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default="/data/dataset/")
    parser.add_argument('--write-dir', default="/data/dataset/")
    args = parser.parse_args()

    main(args)
