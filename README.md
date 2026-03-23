# P2-ETF-MAGAT-ENGINE

**Multi-head Attention Graph Asset Transformer — Supervised ETF Signal Engine**

MAGAT replaces DeePM's LSTM encoder with a per-asset MLP encoder and replaces the fixed macro-conditioned graph prior with a fully learned GAT adjacency. Trained end-to-end with Sharpe/EVaR loss — no RL.

---

## Architecture

```
x_asset (B, n_assets, lookback, n_asset_feats)
x_macro (B, lookback, n_macro_feats)
        ↓
AssetMLPEncoder (flatten window → MLP 2-layer) → (B, A, GAT_HIDDEN=64)
MacroEncoder (linear + mean pool)              → (B, 32)
GATEncoder (multi-head attention, 2 layers)    → (B, A, 64)
  — learns cross-asset adjacency purely from data —
  — no pre-specified macro conditioning —
PortfolioHead (MLP 128→64→A, softmax)         → weights (B, A)
```

**Key differences from DeePM:**
- MLP encoder (not LSTM) — simpler, faster, no vanishing gradients
- Fully learned GAT adjacency (not macro-conditioned graph prior)
- Same Sharpe/EVaR loss, same fixed split + shrinking windows

---

## ETF Universe

### Option A — Fixed Income / Alternatives (benchmark: AGG)
TLT · LQD · HYG · VNQ · GLD · SLV · PFF · MBB

### Option B — Equity Sectors (benchmark: SPY)
SPY · QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME

---

## Setup

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | HuggingFace write token |
| `HF_DATASET_REPO` | `P2SAMAPA/p2-etf-deepm-data` |
| `HF_MODELS_REPO` | `P2SAMAPA/p2-etf-magat-models` |

---

## Disclaimer

Research and educational purposes only. Not financial advice.
