import torch
import torch.nn as nn
import torch.nn.functional as F

from cdvae.pl_modules.embeddings import MAX_ATOMIC_NUM
from cdvae.pl_modules.gemnet.gemnet import GemNetT


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
        )
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, z, data):
        """
        args:
            z: (N_cryst, num_latent)
            data: a torch geometric data object
        returns:
            atom_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        h, pred_cart_coord_diff = self.gemnet(z=z, data=data)
        pred_atom_types = self.fc_atom(h)

        return pred_cart_coord_diff, pred_atom_types
