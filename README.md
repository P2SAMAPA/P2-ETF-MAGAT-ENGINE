# P2-ETF-MAGAT-ENGINE

**Multi-Agent Graph Attention DRL ETF Signal Engine**

Each ETF has its own DQN agent. A GAT layer propagates cross-asset information before each agent decides to be active or inactive. A meta-agent selects the final ETF from active agents. All trading incurs a **15 bps cost** to discourage over-trading.

---

## Architecture

```
State: LOOKBACK-day OHLCV + macro context
           ↓
GATEncoder (multi-head graph attention, 2 layers)
           → node_emb (n_assets, 64)
           ↓
AgentNetwork × n_assets (Dueling DQN)
           → Q(inactive), Q(active) per asset
           ↓
MetaAgent (node_emb + macro → confidence score)
           → pick = argmax(meta_score | active)
```

**Reward function:**
```
reward = log_return(held_ETF) - 0.0015 × I(ETF changed)
```

15 bps trading cost applied whenever the held ETF changes, discouraging daily flipping.

---

## ETF Universe

### Option A — Fixed Income / Alternatives (benchmark: AGG)
TLT · LQD · HYG · VNQ · GLD · SLV · PFF · MBB

### Option B — Equity Sectors (benchmark: SPY)
SPY · QQQ · XLK · XLF · XLE · XLV · XLI · XLY · XLP · XLU · GDX · XME

---

## Training

- Fixed split: 70% train / 15% val / 15% test
- Shrinking windows: 8 windows (2008→2024 down to 2022→2024)
- N_EPISODES=10, GAMMA=0.99, epsilon-greedy exploration
- Dueling DQN with soft target updates (TAU=0.005)
- Replay buffer size: 10,000

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
