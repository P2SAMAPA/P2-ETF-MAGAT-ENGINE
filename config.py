# config.py — P2-ETF-MAGAT-ENGINE
# Multi-Agent Graph Attention DRL Engine

import os

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-deepm-data")
HF_MODELS_REPO  = os.environ.get("HF_MODELS_REPO",  "P2SAMAPA/p2-etf-magat-models")

# ── Data ───────────────────────────────────────────────────────────────────────
FILE_MASTER = "data/master.parquet"

# ── ETF Universes ──────────────────────────────────────────────────────────────
FI_ETFS = [
    "TLT", "LQD", "HYG", "VNQ",
    "GLD", "SLV", "PFF", "MBB",
]
FI_BENCHMARK = "AGG"

EQ_ETFS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
]
EQ_BENCHMARK = "SPY"

# ── Macro features (core 5) ────────────────────────────────────────────────────
MACRO_VARS = ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX", "DTB3"]

# ── State space ────────────────────────────────────────────────────────────────
LOOKBACK      = 20    # shorter lookback for RL (faster environment steps)

# ── Trading cost ───────────────────────────────────────────────────────────────
# 15 bps one-way per trade (applied when ETF selection changes)
TRADING_COST_BPS = 15
TRADING_COST     = TRADING_COST_BPS / 10000.0   # 0.0015

# ── GAT config ─────────────────────────────────────────────────────────────────
GAT_HIDDEN_DIM = 64
GAT_N_HEADS    = 4
GAT_DROPOUT    = 0.1
GAT_N_LAYERS   = 2

# ── DQN agent config ───────────────────────────────────────────────────────────
DQN_HIDDEN_DIM  = 128
GAMMA           = 0.99
TAU             = 0.005   # soft update for target network
LR_AGENT        = 1e-3
REPLAY_BUFFER   = 10000
BATCH_SIZE_RL   = 64
MIN_REPLAY      = 500
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 0.995

# ── Meta-agent ─────────────────────────────────────────────────────────────────
META_HIDDEN_DIM = 64

# ── Training ───────────────────────────────────────────────────────────────────
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
# TEST = remaining 15%

N_EPISODES      = 10      # passes through training data
MAX_STEPS_EP    = 2000    # max steps per episode (days)
WARMUP_EPISODES = 2       # pure exploration before learning

TRAIN_END  = "2024-12-31"
LIVE_START = "2025-01-01"

# ── Shrinking windows ──────────────────────────────────────────────────────────
WINDOWS = [
    {"id": 1, "start": "2008-01-01"},
    {"id": 2, "start": "2010-01-01"},
    {"id": 3, "start": "2012-01-01"},
    {"id": 4, "start": "2014-01-01"},
    {"id": 5, "start": "2016-01-01"},
    {"id": 6, "start": "2018-01-01"},
    {"id": 7, "start": "2020-01-01"},
    {"id": 8, "start": "2022-01-01"},
]

# ── Local dirs ─────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"
