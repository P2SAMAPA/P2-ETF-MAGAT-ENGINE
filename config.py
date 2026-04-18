# config.py — P2-ETF-MAGAT-ENGINE
# Supervised GAT + Portfolio Head engine (no RL)
# GAT replaces LSTM encoder from DeePM
# Dynamic adjacency replaces fixed macro graph prior

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
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "IWM", "IWF", "XSD", "XBI", 
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
]
EQ_BENCHMARK = "SPY"

# ── Macro features ─────────────────────────────────────────────────────────────
MACRO_VARS = ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX", "DTB3"]

# ── Sequence ───────────────────────────────────────────────────────────────────
LOOKBACK = 60    # trading days per sample

# ── GAT config ─────────────────────────────────────────────────────────────────
# Each asset's LOOKBACK-day feature vector is encoded by a per-asset MLP,
# then GAT propagates cross-asset information
GAT_HIDDEN_DIM  = 64
GAT_N_HEADS     = 4
GAT_N_LAYERS    = 2
GAT_DROPOUT     = 0.1

# ── Macro encoder ──────────────────────────────────────────────────────────────
MACRO_HIDDEN_DIM = 32

# ── Portfolio head ─────────────────────────────────────────────────────────────
PORT_HIDDEN_DIM = 128

# ── Training ───────────────────────────────────────────────────────────────────
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15

BATCH_SIZE   = 64
MAX_EPOCHS   = 150
PATIENCE     = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.2

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
