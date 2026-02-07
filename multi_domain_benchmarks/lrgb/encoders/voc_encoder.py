"""
VOC superpixels encoders: VOCNodeEncoder, VOCEdgeEncoder. Used by helpers.encoders for Pascal VOC.
"""
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

VOC_node_input_dim = 14


class VOCNodeEncoder(torch.nn.Module):
    """Linear encoder for VOC node features (14-dim to emb_dim)."""
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, x, pestat):
        #batch.x = self.encoder(batch.x)

        return self.encoder(x)

register_node_encoder('VOCNode', VOCNodeEncoder)


class VOCEdgeEncoder(torch.nn.Module):
    """Linear encoder for VOC edge features (2-dim to emb_dim)."""
    def __init__(self, emb_dim):
        super().__init__()

        VOC_edge_input_dim = 2# if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, edge_attr):
        return self.encoder(edge_attr)