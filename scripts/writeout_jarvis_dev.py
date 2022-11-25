""" Isolates a subset of Jarvis data for development work """

import pandas as pd


def main():
    print("reading csv...")
    df = pd.read_csv("~/Projects/mila/cdvae/data/jarvis/train.csv", index_col="dataset_id")#, nrows=1000))
    df = df[df["dataset_name"]=="DFT3D"]
    df = df[df["num_atoms"] <= 25]
    df = df.sample(1000)
    df.to_csv("~/Projects/mila/cdvae/data/jarvis_dev/train.csv")

    df = pd.read_csv("~/Projects/mila/cdvae/data/jarvis/val.csv", index_col="dataset_id")#, nrows=500)
    df = df[df["dataset_name"]=="DFT3D"]
    df = df[df["num_atoms"] <= 25]
    df = df.sample(1000)
    df.to_csv("~/Projects/mila/cdvae/data/jarvis_dev/val.csv")

    df = pd.read_csv("~/Projects/mila/cdvae/data/jarvis/test.csv", index_col="dataset_id")#, nrows=250)
    df = df[df["dataset_name"]=="DFT3D"]
    df = df[df["num_atoms"] <= 25]
    df = df.sample(1000)
    df.to_csv("~/Projects/mila/cdvae/data/jarvis_dev/test.csv")


if __name__ == "__main__":
    main()
