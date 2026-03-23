# model.py — MAGAT Multi-Agent Graph Attention Network
#
# Architecture:
#   State: per-asset OHLCV+return window + macro context
#       ↓
#   GATEncoder   — multi-head graph attention across all assets
#       → node embeddings (n_assets, GAT_HIDDEN_DIM)
#   AgentNetwork — per-asset DQN head: Q(s,a) for a ∈ {inactive, active}
#       → Q-values (n_assets, 2)
#   MetaAgent    — selects final ETF from active agents by highest Q-value
#       → pick (scalar)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ── GAT Layer ─────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Single graph attention layer.
    Input:  x (n_assets, in_dim)
    Output: x (n_assets, out_dim * n_heads)
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat

        self.W    = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, n_heads, 2 * out_dim))
        self.drop = nn.Dropout(dropout)
        self.act  = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        Wx = self.W(x).view(N, self.n_heads, self.out_dim)  # (N, H, D)

        # Attention coefficients
        src = Wx.unsqueeze(1).expand(N, N, self.n_heads, self.out_dim)
        tgt = Wx.unsqueeze(0).expand(N, N, self.n_heads, self.out_dim)
        cat = torch.cat([src, tgt], dim=-1)                  # (N, N, H, 2D)

        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        e   = self.act((cat * self.attn).sum(-1))            # (N, N, H)
        a   = F.softmax(e, dim=1)                            # (N, N, H)
        a   = self.drop(a)

        # Aggregate
        out = (a.unsqueeze(-1) * Wx.unsqueeze(0)).sum(1)     # (N, H, D)

        if self.concat:
            return out.view(N, self.n_heads * self.out_dim)  # (N, H*D)
        else:
            return out.mean(1)                               # (N, D)


class GATEncoder(nn.Module):
    """
    Stacked GAT layers for cross-asset information aggregation.
    Input:  asset_states (n_assets, state_dim)
    Output: node_emb     (n_assets, GAT_HIDDEN_DIM)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        layers = []
        in_d   = state_dim
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            out_d   = hidden_dim
            layers.append(GATLayer(
                in_dim=in_d,
                out_dim=out_d // (1 if is_last else n_heads),
                n_heads=n_heads,
                dropout=dropout,
                concat=not is_last,
            ))
            in_d = out_d if is_last else out_d
        self.layers = nn.ModuleList(layers)
        self.norm   = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.elu(layer(x))
        return self.norm(x)


# ── Per-Asset DQN Agent ────────────────────────────────────────────────────────

class AgentNetwork(nn.Module):
    """
    Dueling DQN head for a single asset agent.
    Input:  node_emb (GAT_HIDDEN_DIM,) — from GAT encoder
    Output: Q-values (2,) — Q(inactive), Q(active)

    Uses dueling architecture: Q = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        # Value stream
        self.value  = nn.Linear(hidden_dim // 2, 1)
        # Advantage stream
        self.adv    = nn.Linear(hidden_dim // 2, 2)  # 2 actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.shared(x)
        V   = self.value(h)                          # (1,)
        A   = self.adv(h)                            # (2,)
        Q   = V + A - A.mean(dim=-1, keepdim=True)  # dueling
        return Q


# ── Meta-Agent ─────────────────────────────────────────────────────────────────

class MetaAgent(nn.Module):
    """
    Selects the final ETF from active agents.
    Takes all agent Q-values + GAT embeddings + macro context.
    Outputs a confidence score per agent (used to rank active agents).
    """

    def __init__(self, gat_dim: int, n_macro: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gat_dim + n_macro, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_emb: torch.Tensor,
                macro_ctx: torch.Tensor) -> torch.Tensor:
        """
        node_emb:  (n_assets, gat_dim)
        macro_ctx: (n_macro,)
        Returns:   (n_assets,) confidence scores
        """
        macro_exp = macro_ctx.unsqueeze(0).expand(node_emb.size(0), -1)
        combined  = torch.cat([node_emb, macro_exp], dim=-1)
        return self.net(combined).squeeze(-1)             # (n_assets,)


# ── Full MAGAT Model ───────────────────────────────────────────────────────────

class MAGAT(nn.Module):
    """
    Full Multi-Agent Graph Attention model.
    Holds all agent networks + GAT encoder + meta-agent.
    Used for inference (selecting the daily ETF pick).
    """

    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        n_macro: int,
        gat_hidden: int  = 64,
        gat_heads: int   = 4,
        gat_layers: int  = 2,
        dqn_hidden: int  = 128,
        meta_hidden: int = 64,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets

        self.gat_encoder = GATEncoder(
            state_dim=state_dim,
            hidden_dim=gat_hidden,
            n_heads=gat_heads,
            n_layers=gat_layers,
            dropout=dropout,
        )
        self.agents = nn.ModuleList([
            AgentNetwork(gat_hidden, dqn_hidden)
            for _ in range(n_assets)
        ])
        self.meta_agent = MetaAgent(gat_hidden, n_macro, meta_hidden)

    def forward(self, asset_states: torch.Tensor,
                macro_ctx: torch.Tensor) -> dict:
        """
        asset_states: (n_assets, state_dim)
        macro_ctx:    (n_macro,)

        Returns dict with:
            node_emb:   (n_assets, gat_hidden)
            q_values:   (n_assets, 2)
            active:     (n_assets,) bool — action=1
            meta_scores:(n_assets,) confidence
            pick:       int — index of selected ETF
            weights:    (n_assets,) softmax of meta scores for active agents
        """
        node_emb    = self.gat_encoder(asset_states)       # (A, gat_hidden)

        q_values    = torch.stack([
            agent(node_emb[i]) for i, agent in enumerate(self.agents)
        ])                                                  # (A, 2)

        actions     = q_values.argmax(dim=-1)              # (A,) 0=inactive, 1=active
        active_mask = actions.bool()                        # (A,)

        meta_scores = self.meta_agent(node_emb, macro_ctx) # (A,)

        # Among active agents, pick highest meta score
        masked_scores = meta_scores.clone()
        masked_scores[~active_mask] = -float("inf")

        # Softmax over active agents for weights
        active_any = active_mask.any()
        if active_any:
            weights = F.softmax(masked_scores, dim=0)
            pick    = int(masked_scores.argmax())
        else:
            # All inactive — fall back to highest Q(active) overall
            weights = F.softmax(q_values[:, 1], dim=0)
            pick    = int(q_values[:, 1].argmax())

        return {
            "node_emb":    node_emb,
            "q_values":    q_values,
            "active":      active_mask,
            "meta_scores": meta_scores,
            "pick":        pick,
            "weights":     weights,
        }


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak soft update for target network."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
