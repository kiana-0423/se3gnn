from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn


try:  # Torch Geometric is optional; fall back to duck typing when unavailable.
    from torch_geometric.data import Data
    from torch_geometric.typing import Adj
except ImportError:  # pragma: no cover - optional dependency
    Data = object
    Adj = Tensor


@dataclass
class SE3TransformerConfig:
    """Configuration container for the SE(3)-Transformer skeleton."""

    atom_feature_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    radial_basis_size: int = 32
    cutoff: float = 5.0
    dropout: float = 0.0
    layer_norm: bool = True


class RadialBasis(nn.Module):
    """
    Simple Gaussian radial basis encoder.

    The majority of equivariant models use a smooth radial expansion (Bessel,
    Gaussian, or a learnable basis).  This class keeps things simple while
    providing a clear place to slot in a more sophisticated basis later.
    """

    def __init__(self, cutoff: float, num_basis: int) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        widths = torch.full_like(centers, 0.5 * (centers[1] - centers[0] + 1e-2))
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        distances = torch.clamp(distances, max=self.cutoff)
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff / self.widths) ** 2)


class EquivariantSelfAttention(nn.Module):
    """
    Placeholder self-attention block.

    The module currently performs a simple message passing update.  Project
    specific implementations can extend this class to incorporate tensor
    products or spherical harmonics while keeping the public interface intact.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        radial_basis: RadialBasis,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.radial_basis = radial_basis
        self.dropout = nn.Dropout(dropout)

        edge_input_dim = 2 * hidden_dim + radial_basis.centers.numel()
        self.message_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_features: Tensor,
        positions: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        src, dst = edge_index
        relative_vec = positions[dst] - positions[src]
        distances = torch.norm(relative_vec, dim=-1)

        if edge_attr is None:
            edge_attr = self.radial_basis(distances)

        messages = torch.cat(
            (node_features[src], node_features[dst], edge_attr),
            dim=-1,
        )
        messages = self.message_mlp(messages)
        messages = self.dropout(messages)

        updates = torch.zeros_like(node_features)
        updates.index_add_(0, dst, messages)

        degree = torch.zeros(
            node_features.shape[0],
            device=node_features.device,
            dtype=node_features.dtype,
        )
        degree.index_add_(0, dst, torch.ones_like(distances))
        degree = degree.clamp_min(1.0).unsqueeze(-1)

        updates = updates / degree
        updates = self.output_projection(updates)

        # Positional updates are model dependant.  Return None for now.
        return updates, None


class SE3TransformerLayer(nn.Module):
    """One residual block of the SE(3)-Transformer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        radial_basis: RadialBasis,
        dropout: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.attention = EquivariantSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            radial_basis=radial_basis,
            dropout=dropout,
        )
        self.post_attn_norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.post_ffn_norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(
        self,
        node_features: Tensor,
        positions: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        delta_features, delta_pos = self.attention(
            node_features=node_features,
            positions=positions,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        node_features = node_features + delta_features
        node_features = self.post_attn_norm(node_features)

        node_features = node_features + self.ffn(node_features)
        node_features = self.post_ffn_norm(node_features)

        if delta_pos is not None:
            positions = positions + delta_pos

        return node_features, positions


class SE3Transformer(nn.Module):
    """
    High-level SE(3)-Transformer wrapper.

    The forward method expects a `torch_geometric.data.Data` object with fields:
      - `x`: Node features (num_nodes, atom_feature_dim)
      - `pos`: Cartesian coordinates (num_nodes, 3)
      - `edge_index`: Edge list in COO format (2, num_edges)
      - `edge_attr` (optional): Pre-computed edge features
    """

    def __init__(self, config: SE3TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.node_embedding = nn.Linear(config.atom_feature_dim, config.hidden_dim)
        self.radial_basis = RadialBasis(
            cutoff=config.cutoff,
            num_basis=config.radial_basis_size,
        )
        self.layers = nn.ModuleList(
            [
                SE3TransformerLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    radial_basis=self.radial_basis,
                    dropout=config.dropout,
                    layer_norm=config.layer_norm,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        batch: Data,
    ) -> Tensor:
        node_features = self.node_embedding(batch.x)
        positions = batch.pos
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)

        for layer in self.layers:
            node_features, positions = layer(
                node_features=node_features,
                positions=positions,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        per_node_energy = self.readout(node_features)
        energy = torch.sum(per_node_energy, dim=0)
        return energy.squeeze()
