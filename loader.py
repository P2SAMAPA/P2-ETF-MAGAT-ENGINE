# loader.py — Loads data from shared p2-etf-deepm-data HF dataset

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import config as cfg


def _load_parquet(filename: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=cfg.HF_TOKEN or None,
        force_download=True,
    )
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def load_master() -> pd.DataFrame:
    print("[loader] Loading master dataset...")
    df = _load_parquet(cfg.FILE_MASTER)
    print(f"[loader] Master: {df.shape}, "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def get_option_data(option: str, master: pd.DataFrame) -> dict:
    tickers   = cfg.FI_ETFS   if option == "A" else cfg.EQ_ETFS
    benchmark = cfg.FI_BENCHMARK if option == "A" else cfg.EQ_BENCHMARK

    # ── Log returns ──────────────────────────────────────────────────────────
    logret_cols = [f"{t}_logret" for t in tickers
                   if f"{t}_logret" in master.columns]
    returns = master[logret_cols].copy().ffill().fillna(0.0)
    returns.columns = [c.replace("_logret", "") for c in returns.columns]

    # ── OHLCV features per asset ──────────────────────────────────────────────
    # Build (T, n_assets, n_feats): Open, High, Low, Close, Volume, logret
    feat_list = []
    for t in tickers:
        cols = {}
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            col = f"{t}_{field}"
            if col in master.columns:
                cols[field] = master[col]
            else:
                cols[field] = pd.Series(0.0, index=master.index)
        cols["logret"] = returns[t] if t in returns.columns else \
                         pd.Series(0.0, index=master.index)

        feat_df = pd.DataFrame(cols, index=master.index).ffill().fillna(0.0)
        feat_list.append(feat_df.values)   # (T, 6)

    # Stack to (T, n_assets, n_feats)
    ohlcv_feat = np.stack(feat_list, axis=1).astype(np.float32)

    # Normalise each feature across time (z-score)
    for f in range(ohlcv_feat.shape[-1]):
        mu  = ohlcv_feat[:, :, f].mean()
        std = ohlcv_feat[:, :, f].std() + 1e-8
        ohlcv_feat[:, :, f] = (ohlcv_feat[:, :, f] - mu) / std

    # ── Macro features ────────────────────────────────────────────────────────
    macro_cols = [c for c in cfg.MACRO_VARS if c in master.columns]
    macro_df   = master[macro_cols].copy().ffill().fillna(0.0)
    # z-score normalise
    macro_arr  = macro_df.values.astype(np.float32)
    macro_arr  = (macro_arr - macro_arr.mean(0)) / (macro_arr.std(0) + 1e-8)

    # ── Cash rate ─────────────────────────────────────────────────────────────
    cash_rate = master["TBILL_daily"].fillna(0.0).values.astype(np.float32) \
                if "TBILL_daily" in master.columns \
                else np.zeros(len(master), dtype=np.float32)

    # ── Benchmark returns ─────────────────────────────────────────────────────
    bench_ret = master[f"{benchmark}_ret"].fillna(0.0).values.astype(np.float32) \
                if f"{benchmark}_ret" in master.columns \
                else np.zeros(len(master), dtype=np.float32)

    # ── Splits ────────────────────────────────────────────────────────────────
    n       = len(master)
    n_train = int(n * cfg.TRAIN_SPLIT)
    n_val   = int(n * cfg.VAL_SPLIT)
    n_test  = n - n_train - n_val

    idx = master.index

    print(f"[loader] Option {option} ({len(tickers)} ETFs): "
          f"{n} days | train={n_train} | val={n_val} | test={n_test}")
    print(f"[loader] Test start: {idx[n_train + n_val].date()}")

    ret_arr = returns[tickers].values.astype(np.float32)

    return {
        "option":     option,
        "tickers":    tickers,
        "benchmark":  benchmark,
        "index":      idx,
        # full arrays
        "returns":    ret_arr,
        "ohlcv_feat": ohlcv_feat,
        "macro_feat": macro_arr,
        "cash_rate":  cash_rate,
        "bench_ret":  bench_ret,
        # split indices
        "n_train":    n_train,
        "n_val":      n_val,
        "n_test":     n_test,
        "test_start": str(idx[n_train + n_val].date()),
        # split slices
        "train_ret":  ret_arr[:n_train],
        "val_ret":    ret_arr[n_train:n_train + n_val],
        "test_ret":   ret_arr[n_train + n_val:],
        "train_ohlcv":ohlcv_feat[:n_train],
        "val_ohlcv":  ohlcv_feat[n_train:n_train + n_val],
        "test_ohlcv": ohlcv_feat[n_train + n_val:],
        "train_macro":macro_arr[:n_train],
        "val_macro":  macro_arr[n_train:n_train + n_val],
        "test_macro": macro_arr[n_train + n_val:],
        "n_asset_feats": ohlcv_feat.shape[-1],
        "n_macro_feats": macro_arr.shape[-1],
    }


def get_window_data(data: dict, start: str, end: str) -> dict:
    """Slice data to a specific date window."""
    idx   = data["index"]
    mask  = (idx >= start) & (idx <= end)
    oos_m = idx >= cfg.LIVE_START
    n_w   = mask.sum()
    n_oos = oos_m.sum()

    return {
        **data,
        "returns":    data["returns"][mask],
        "ohlcv_feat": data["ohlcv_feat"][mask],
        "macro_feat": data["macro_feat"][mask],
        "n_train":    n_w,
        "oos_ret":    data["returns"][oos_m],
        "oos_ohlcv":  data["ohlcv_feat"][oos_m],
        "oos_macro":  data["macro_feat"][oos_m],
        "n_oos":      n_oos,
    }
