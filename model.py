# model.py — MAGAT Supervised Architecture
#
# Key difference from DeePM:
#   DeePM: LSTM encoder per asset → fixed macro graph prior → portfolio head
#   MAGAT: MLP encoder per asset → LEARNED GAT adjacency → portfolio head
#
# The GAT layer learns which assets attend to which other assets
# purely from the data — no pre-specified macro conditioning.
# This makes the cross-asset graph fully data-driven.
#
# Architecture:
#   x_asset (B, n_assets, lookback, n_asset_feats)
#   x_macro (B, lookback, n_macro_feats)
#       ↓
#   AssetMLPEncoder  — per-asset MLP on flattened window → (B, A, GAT_HIDDEN)
#   MacroEncoder     — linear + mean pool → (B, MACRO_HIDDEN)
#   GATLayer × N     — learned cross-asset graph attention → (B, A, GAT_HIDDEN)
#   PortfolioHead    — MLP → softmax over n_assets → weights (B, A)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ── Asset MLP Encoder ──────────────────────────────────────────────────────────

class AssetMLPEncoder(nn.Module):
    """
    Encodes each asset's LOOKBACK-day feature window independently.
    Flattens (lookback × n_feats) → MLP → hidden embedding.

    Input:  (B, n_assets, lookback, n_feats)
    Output: (B, n_assets, hidden_dim)
    """

    def __init__(self, lookback: int, n_feats: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        in_dim = lookback * n_feats
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, A, L, F = x.shape
        x = x.view(B, A, L * F)          # flatten lookback × feats
        x = self.net(x)                   # (B, A, hidden_dim)
        return self.norm(x)


# ── Macro Encoder ──────────────────────────────────────────────────────────────

class MacroEncoder(nn.Module):
    """
    Encodes macro time series into a context vector.
    Input:  (B, L, n_macro_feats)
    Output: (B, macro_hidden_dim)
    """

    def __init__(self, n_macro_feats: int, hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_macro_feats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)         # (B, L, hidden_dim)
        x = x.mean(dim=1)        # mean pool over time → (B, hidden_dim)
        return self.norm(x)


# ── GAT Layer ─────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Multi-head graph attention layer.
    Learns which assets attend to which — fully data-driven adjacency.

    Input:  x (B, n_assets, in_dim)
    Output: x (B, n_assets, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat
        self.head_dim = out_dim // n_heads if concat else out_dim

        self.W    = nn.Linear(in_dim, self.head_dim * n_heads, bias=False)
        self.attn = nn.Parameter(torch.empty(1, n_heads, 2 * self.head_dim))
        self.drop = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D    = self.n_heads, self.head_dim

        Wx = self.W(x).view(B, N, H, D)                      # (B, N, H, D)

        # Pairwise attention scores
        src = Wx.unsqueeze(2).expand(B, N, N, H, D)          # (B, N, N, H, D)
        tgt = Wx.unsqueeze(1).expand(B, N, N, H, D)
        cat = torch.cat([src, tgt], dim=-1)                   # (B, N, N, H, 2D)

        e   = self.leaky((cat * self.attn).sum(-1))           # (B, N, N, H)
        a   = F.softmax(e, dim=2)                             # (B, N, N, H)
        a   = self.drop(a)

        # Aggregate
        out = (a.unsqueeze(-1) * Wx.unsqueeze(1)).sum(2)      # (B, N, H, D)

        if self.concat:
            out = out.view(B, N, H * D)                       # (B, N, H*D)
        else:
            out = out.mean(2)                                  # (B, N, D)

        return F.elu(out)


class GATEncoder(nn.Module):
    """
    Stacked GAT layers.
    Input:  (B, n_assets, in_dim)
    Output: (B, n_assets, GAT_HIDDEN_DIM)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            out_d   = hidden_dim
            layers.append(GATLayer(
                in_dim=d,
                out_dim=out_d,
                n_heads=n_heads,
                dropout=dropout,
                concat=not is_last,
            ))
            d = out_d
        self.layers = nn.ModuleList(layers)
        self.norm   = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ── Portfolio Head ─────────────────────────────────────────────────────────────

class PortfolioHead(nn.Module):
    """
    Maps GAT embeddings + macro context to portfolio weights.
    Input:  graph_emb (B, A, GAT_HIDDEN_DIM)
            macro_ctx (B, MACRO_HIDDEN_DIM)
    Output: weights   (B, A)
    """

    def __init__(self, n_assets: int, gat_hidden: int,
                 macro_hidden: int, hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        combined = gat_hidden + macro_hidden
        self.head = nn.Sequential(
            nn.Linear(combined, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_assets),
        )

    def forward(self, graph_emb: torch.Tensor,
                macro_ctx: torch.Tensor) -> torch.Tensor:
        pool    = graph_emb.mean(dim=1)                       # (B, gat_hidden)
        combined = torch.cat([pool, macro_ctx], dim=-1)       # (B, gat+macro)
        return F.softmax(self.head(combined), dim=-1)         # (B, A)


# ── Full MAGAT Model ───────────────────────────────────────────────────────────

class MAGAT(nn.Module):
    """
    MAGAT: Multi-head Attention Graph Asset Transformer (supervised).

    Replaces DeePM's LSTM encoder with MLP and fixed graph prior
    with learned GAT adjacency. Trained end-to-end with Sharpe/EVaR loss.
    """

    def __init__(
        self,
        n_assets: int,
        n_asset_feats: int,
        n_macro_feats: int,
        lookback: int        = 60,
        gat_hidden: int      = 64,
        gat_heads: int       = 4,
        gat_layers: int      = 2,
        macro_hidden: int    = 32,
        port_hidden: int     = 128,
        dropout: float       = 0.2,
    ):
        super().__init__()
        self.n_assets = n_assets

        self.asset_encoder = AssetMLPEncoder(
            lookback=lookback,
            n_feats=n_asset_feats,
            hidden_dim=gat_hidden,
            dropout=dropout,
        )
        self.macro_encoder = MacroEncoder(
            n_macro_feats=n_macro_feats,
            hidden_dim=macro_hidden,
            dropout=dropout,
        )
        self.gat_encoder = GATEncoder(
            in_dim=gat_hidden,
            hidden_dim=gat_hidden,
            n_heads=gat_heads,
            n_layers=gat_layers,
            dropout=dropout,
        )
        self.portfolio_head = PortfolioHead(
            n_assets=n_assets,
            gat_hidden=gat_hidden,
            macro_hidden=macro_hidden,
            hidden_dim=port_hidden,
            dropout=dropout,
        )

    def forward(self, x_asset: torch.Tensor,
                x_macro: torch.Tensor) -> torch.Tensor:
        """
        x_asset: (B, n_assets, lookback, n_asset_feats)
        x_macro: (B, lookback, n_macro_feats)
        Returns: weights (B, n_assets)
        """
        asset_emb = self.asset_encoder(x_asset)    # (B, A, gat_hidden)
        macro_ctx = self.macro_encoder(x_macro)    # (B, macro_hidden)
        graph_emb = self.gat_encoder(asset_emb)    # (B, A, gat_hidden)
        weights   = self.portfolio_head(graph_emb, macro_ctx)  # (B, A)
        return weights


# ── Loss Functions ─────────────────────────────────────────────────────────────

def sharpe_loss(weights: torch.Tensor, returns: torch.Tensor,
                cash_rate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    port_ret = (weights * returns).sum(dim=1)
    excess   = port_ret - cash_rate
    return -(excess.mean() / (excess.std() + eps)) * math.sqrt(252)


def evar_loss(weights: torch.Tensor, returns: torch.Tensor,
              cash_rate: torch.Tensor, beta: float = 0.95,
              eps: float = 1e-6) -> torch.Tensor:
    port_ret = (weights * returns).sum(dim=1)
    excess   = port_ret - cash_rate
    mean_ret = excess.mean()
    t        = torch.tensor(1.0)
    evar_val = t * torch.log(
        torch.mean(torch.exp(-excess / (t + eps))) + eps
    ) + t * math.log(1.0 / (1.0 - beta))
    return evar_val - mean_ret * 0.5


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
