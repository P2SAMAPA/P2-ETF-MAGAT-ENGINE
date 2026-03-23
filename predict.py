# predict.py — MAGAT daily signal generation
#
# Usage:
#   python predict.py --option both

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch

import config as cfg
import loader
from model import MAGAT

DEVICE = torch.device("cpu")


def next_trading_day(from_date: str = None) -> str:
    nyse = mcal.get_calendar("NYSE")
    base = pd.Timestamp(from_date) if from_date else pd.Timestamp.today()
    schedule = nyse.schedule(
        start_date=base.strftime("%Y-%m-%d"),
        end_date=(base + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    days = mcal.date_range(schedule, frequency="1D").normalize().tz_localize(None)
    future = [d for d in days if d > base]
    return str(future[0].date()) if future else \
           str((base + pd.Timedelta(days=1)).date())


def _load_magat(model_path: str, meta: dict) -> MAGAT:
    cfg_m = meta.get("config", {})
    model = MAGAT(
        n_assets=meta["n_assets"],
        state_dim=meta["n_asset_feats"] * cfg_m.get("lookback", cfg.LOOKBACK),
        n_macro=meta["n_macro_feats"],
        gat_hidden=cfg_m.get("gat_hidden", cfg.GAT_HIDDEN_DIM),
        gat_heads=cfg_m.get("gat_heads", cfg.GAT_N_HEADS),
        gat_layers=cfg_m.get("gat_layers", cfg.GAT_N_LAYERS),
        dqn_hidden=cfg_m.get("dqn_hidden", cfg.DQN_HIDDEN_DIM),
        meta_hidden=cfg.META_HIDDEN_DIM,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def _get_last_state(data: dict, meta: dict) -> tuple:
    """Get last LOOKBACK-day state for inference."""
    lookback = meta.get("config", {}).get("lookback", cfg.LOOKBACK)
    ohlcv    = data["ohlcv_feat"]
    macro    = data["macro_feat"]

    # Last window
    window   = ohlcv[-lookback:]                          # (L, A, F)
    asset_states = window.transpose(1, 0, 2).reshape(
        data["n_assets"], lookback * data["n_asset_feats"]
    ).astype(np.float32)
    macro_state = macro[-1].astype(np.float32)

    last_date = str(data["index"][-1].date())
    return asset_states, macro_state, last_date


def generate_signal(option: str, data: dict) -> dict:
    print(f"\n[predict] Generating fixed split signal for Option {option}...")

    model_path = os.path.join(cfg.MODELS_DIR, f"magat_option{option}_best.pt")
    meta_path  = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")

    if not os.path.exists(model_path):
        print(f"  No model found — run train.py first.")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    model = _load_magat(model_path, meta)
    asset_states, macro_state, last_date = _get_last_state(data, meta)
    signal_date = next_trading_day(last_date)

    with torch.no_grad():
        out = model(
            torch.tensor(asset_states),
            torch.tensor(macro_state),
        )

    tickers    = data["tickers"]
    pick       = tickers[out["pick"]]
    conviction = float(out["weights"][out["pick"]])
    weights    = {tickers[i]: round(float(out["weights"][i]), 4)
                  for i in range(len(tickers))}
    active     = {tickers[i]: bool(out["active"][i])
                  for i in range(len(tickers))}

    rc = data["macro_feat"][-1]
    macro_keys = cfg.MACRO_VARS[:data["n_macro_feats"]]
    regime_context = {k: round(float(rc[i]), 3)
                      for i, k in enumerate(macro_keys)}

    print(f"  Option {option}: {pick} (conviction={conviction:.1%}) | "
          f"active_agents={sum(active.values())}/{len(tickers)}")

    return {
        "option":          option,
        "mode":            "fixed_split",
        "option_name":     "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":     signal_date,
        "last_data_date":  last_date,
        "generated_at":    datetime.utcnow().isoformat(),
        "pick":            pick,
        "conviction":      round(conviction, 4),
        "weights":         weights,
        "active_agents":   active,
        "regime_context":  regime_context,
        "trained_at":      meta.get("trained_at", ""),
        "test_ann_return": meta.get("test_ann_return", 0),
        "test_ann_vol":    meta.get("test_ann_vol", 0),
        "test_sharpe":     meta.get("test_sharpe", 0),
        "test_hit_rate":   meta.get("test_hit_rate", 0),
        "test_n_trades":   meta.get("test_n_trades", 0),
        "test_start":      meta.get("test_start", ""),
        "trading_cost_bps":meta.get("trading_cost_bps", cfg.TRADING_COST_BPS),
        "model_n_params":  meta.get("n_params", 0),
    }


def generate_window_signal(option: str, data: dict) -> dict:
    print(f"\n[predict] Generating window signal for Option {option}...")

    model_path = os.path.join(cfg.MODELS_DIR,
                              f"magat_option{option}_window_best.pt")
    meta_path  = os.path.join(cfg.MODELS_DIR,
                              f"meta_option{option}_window.json")

    if not os.path.exists(model_path):
        print(f"  No window model — run train_windows.py first.")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    model = _load_magat(model_path, meta)
    asset_states, macro_state, last_date = _get_last_state(data, meta)
    signal_date = next_trading_day(last_date)

    with torch.no_grad():
        out = model(
            torch.tensor(asset_states),
            torch.tensor(macro_state),
        )

    tickers    = data["tickers"]
    pick       = tickers[out["pick"]]
    conviction = float(out["weights"][out["pick"]])
    weights    = {tickers[i]: round(float(out["weights"][i]), 4)
                  for i in range(len(tickers))}
    active     = {tickers[i]: bool(out["active"][i])
                  for i in range(len(tickers))}

    print(f"  Option {option} window: {pick} (conviction={conviction:.1%}) | "
          f"Window {meta['winning_window']}: "
          f"{meta['winning_train_start']}→{meta['winning_train_end']}")

    return {
        "option":              option,
        "mode":                "shrinking_window",
        "option_name":         "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":         signal_date,
        "last_data_date":      last_date,
        "generated_at":        datetime.utcnow().isoformat(),
        "pick":                pick,
        "conviction":          round(conviction, 4),
        "weights":             weights,
        "active_agents":       active,
        "trained_at":          meta.get("trained_at", ""),
        "winning_window":      meta.get("winning_window", 0),
        "winning_train_start": meta.get("winning_train_start", ""),
        "winning_train_end":   meta.get("winning_train_end", ""),
        "oos_ann_return":      meta.get("oos_ann_return", 0),
        "oos_ann_vol":         meta.get("oos_ann_vol", 0),
        "oos_sharpe":          meta.get("oos_sharpe", 0),
        "oos_hit_rate":        meta.get("oos_hit_rate", 0),
        "oos_n_trades":        meta.get("oos_n_trades", 0),
        "trading_cost_bps":    meta.get("trading_cost_bps", cfg.TRADING_COST_BPS),
    }


def update_history(signal: dict, option: str) -> None:
    path    = os.path.join(cfg.MODELS_DIR, f"signal_history_{option}.json")
    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    record = {
        "signal_date":   signal["signal_date"],
        "pick":          signal["pick"],
        "conviction":    signal["conviction"],
        "active_agents": sum(signal.get("active_agents", {}).values()),
        "generated_at":  signal["generated_at"],
    }
    if record["signal_date"] not in {r["signal_date"] for r in history}:
        history.append(record)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[predict] History: {len(history)} records for Option {option}")


def save_signals(sig_A=None, sig_B=None, sig_Aw=None, sig_Bw=None):
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    combined = {
        "generated_at":    datetime.utcnow().isoformat(),
        "option_A":        sig_A,
        "option_B":        sig_B,
        "option_A_window": sig_Aw,
        "option_B_window": sig_Bw,
    }
    with open(os.path.join(cfg.MODELS_DIR, "latest_signals.json"), "w") as f:
        json.dump(combined, f, indent=2)

    for sig, name, opt, hist in [
        (sig_A,  "signal_A",        "A", True),
        (sig_B,  "signal_B",        "B", True),
        (sig_Aw, "signal_A_window", "A", False),
        (sig_Bw, "signal_B_window", "B", False),
    ]:
        if sig:
            with open(os.path.join(cfg.MODELS_DIR, f"{name}.json"), "w") as f:
                json.dump(sig, f, indent=2)
            if hist:
                update_history(sig, opt)

    print("[predict] All signals saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    print("[predict] Loading master dataset...")
    master = loader.load_master()

    sig_A = sig_B = sig_Aw = sig_Bw = None

    if args.option in ("A", "both"):
        data_A = loader.get_option_data("A", master)
        sig_A  = generate_signal("A", data_A)
        sig_Aw = generate_window_signal("A", data_A)

    if args.option in ("B", "both"):
        data_B = loader.get_option_data("B", master)
        sig_B  = generate_signal("B", data_B)
        sig_Bw = generate_window_signal("B", data_B)

    save_signals(sig_A, sig_B, sig_Aw, sig_Bw)

    print("\n[predict] Done.")
    for sig, label in [(sig_A, "A fixed"), (sig_B, "B fixed"),
                       (sig_Aw, "A window"), (sig_Bw, "B window")]:
        if sig:
            print(f"  Option {label}: {sig['pick']} on {sig['signal_date']} "
                  f"(conviction={sig['conviction']:.1%})")
