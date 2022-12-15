import os
import pickle
from tqdm import tqdm

import numpy as np

def main():
    pi_dir = "/home/jake/Projects/mila/molecule-representation-tda/data/raw/mp-pt/"
    pi_files = os.listdir(pi_dir)
    pi_files.sort()

    all_keys = []
    all_pis = []
    all_pis_full = []
    for pif in tqdm(pi_files):
        pif = open(os.path.join(pi_dir,pif), "rb")
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

    with open("/home/jake/Projects/mila/cdvae/data/mp/persistence_tensors/all_keys.npy", "wb") as f:
        np.save(f, all_keys)

    with open(os.path.join(root,"cdvae/data/qm9/persistence_tensors/all_pis.npy"), "wb") as f:
        np.save(f, all_pis_full)

    with open("/home/jake/Projects/mila/cdvae/data/mp/persistence_tensors/all_pis_plain.npy", "wb") as f:
        np.save(f, all_pis)


if __name__ == "__main__":
    main()
