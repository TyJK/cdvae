""" Write predictions on test set for a given model.

  Example run command:
  python /home/jake/Projects/mila/cdvae/scripts/write_test_predictions.py --model_ckpt_path /home/jake/Projects/mila/cdvae/hydra/singlerun/2022-12-13/qm9-baseline/ --pred_write_path /home/jake/Projects/mila/cdvae/assets/preds/qm9-baseline.csv

"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from scripts.eval_utils import load_model


SAVE_KEYS = ['dataset_id', 'num_atoms', 'num_atom_loss', 'composition_loss', 'coord_loss', 'type_loss', 'kld_loss', 'property_loss']


def main(model_ckpt_path: os.PathLike, pred_write_path: os.PathLike):
    print("loading model from checkpoint directory...")
    model, loader, cfg = load_model(
        model_path=Path(model_ckpt_path), load_data=True, testing=True
    )

    infer_on_gpu = torch.cuda.is_available()

    if infer_on_gpu:
        model = model.to("cuda")

    print("generating predictions on test set...")
    batch_results = {}
    for batch in tqdm(loader):
        if infer_on_gpu:
            batch = batch.to("cuda")
        with torch.no_grad():
            res = model(batch, teacher_forcing=False, training=False, loss_reduction='none')
            res["num_atoms"] = batch["num_atoms"]
            res["dataset_id"] = np.array(batch["dataset_id"])

            for key in SAVE_KEYS:
                key_vals = batch_results.get(key,[])
                new_val = res[key].cpu().numpy() if key != "dataset_id" else res[key]
                key_vals.append(new_val)
                batch_results[key] = key_vals

    print("... prediction generation finished, aggregating")
    for key in SAVE_KEYS:
        key_vals = batch_results[key]
        key_vals = np.concatenate(key_vals)
        batch_results[key] = key_vals

    batch_results = pd.DataFrame(batch_results)

    print("writing to given path...")
    batch_results.to_csv(pred_write_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt_path', required=True)
    parser.add_argument('--pred_write_path', required=True)
    args = parser.parse_args()

    main(args.model_ckpt_path, args.pred_write_path)
