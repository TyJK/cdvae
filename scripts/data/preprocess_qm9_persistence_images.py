import os
import pickle
from tqdm import tqdm

import numpy as np

def main():
    root = "/home/jake/Projects/mila/"
    pi_dir = os.path.join(root, "molecule-representation-tda/data/raw/qm9-pt/")
    pi_files = os.listdir(pi_dir)
    pi_files.sort()

    all_keys = []
    all_pis_full = []
    for pif in tqdm(pi_files):
        pif = open(os.path.join(pi_dir,pif), "rb")
        pi_dict = pickle.load(pif)
        pif.close()

        keys = list(pi_dict.keys())
        keys.sort()

        pis = []
        for key in keys:
            key_pis = pi_dict[key]
            pis.append(key_pis)

        all_keys.extend(keys)
        all_pis_full.append(np.array(pis))

    all_keys = np.array(all_keys)
    all_pis_full = np.concatenate(all_pis_full).astype(float)
    all_pis_plain = all_pis_full[:,-1,:] # last channel in BCWH tensor is plain PI

    with open(os.path.join(root,"cdvae/data/qm9/persistence_tensors/all_keys.npy"), "wb") as f:
        np.save(f, all_keys)

    with open(os.path.join(root,"cdvae/data/qm9/persistence_tensors/all_pis_full.npy"), "wb") as f:
        np.save(f, all_pis_full)

    with open(os.path.join(root,"cdvae/data/qm9/persistence_tensors/all_pis_plain.npy"), "wb") as f:
        np.save(f, all_pis_plain)


if __name__ == "__main__":
    main()
