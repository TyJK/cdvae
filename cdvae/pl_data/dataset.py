import hydra
import omegaconf
import torch
import numpy as np
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import preprocess_tensors, load_pis


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 pi_dir: ValueNode = None, pi_strategy: ValueNode = False,
                 scaler: ValueNode = None,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name

        self.prop = prop
        self.primitive = primitive
        self.graph_method = graph_method

        self.scaler = scaler

        self.data = pd.read_csv(self.path)
        self.pi_strategy = pi_strategy
        self.pi_data = load_pis(pi_dir, pi_strategy) if pi_strategy else None

        # filter data without pi data if active pi_strategy
        if pi_strategy != 'none':
            data_with_pi = set(self.pi_data.keys())
            self.data = self.data[self.data["dataset_id"].apply(lambda idx: idx in data_with_pi)]
            self.data.index = pd.RangeIndex(len(self.data))
            # reindex pi_data to align with range index of data
            self.pi_data = {idx: self.pi_data[data_id] for (idx, data_id) in self.data["dataset_id"].items()}

        for col in ["coords", "elements"]:
            self.data[col] = self.data[col].apply(eval)  # string-to-list op

        self.data = self.data[["dataset_id","n_atoms","coords","elements", prop]]
        self.data["n_atoms"] = self.data["elements"].apply(len)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data_dict = self.data.loc[index]
        
        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop]).float()

        pi = torch.Tensor(self.pi_data[index]).float() if self.pi_strategy != 'none' else None
       
        data = Data(
            dataset_id=data_dict["dataset_id"],
            coords=torch.Tensor(data_dict["coords"]),
            atom_types=torch.Tensor(data_dict["elements"]).long(),
            n_atoms=torch.Tensor([data_dict["n_atoms"]]).long(),
            persistence_image=pi,
            y=prop
        )

        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, primitive,
                 graph_method, preprocess_workers,
                 **kwargs):
        super().__init__()
        self.primitive = primitive
        self.graph_method = graph_method

        self.data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        self.scaler = None

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index):
        data_dict = self.data[index]


        (coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            coords=torch.Tensor(coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.data)})"



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list

    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )

    scaler = get_scaler_from_data_list(dataset.data, key=dataset.prop)

    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)

    return batch


if __name__ == "__main__":
    main()
