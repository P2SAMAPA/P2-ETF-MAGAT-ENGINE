# train_windows.py — MAGAT shrinking window RL training
#
# Usage:
#   python train_windows.py --option both

import argparse
import copy
import json
import os
import pickle
import shutil
import time
from datetime import datetime

import numpy as np
import torch

import config as cfg
import loader
from model import MAGAT, soft_update, count_parameters
from environment import TradingEnv, ReplayBuffer
from train import update_agents, evaluate, DEVICE

os.makedirs(cfg.MODELS_DIR, exist_ok=True)


def train_window(window: dict, data: dict, option: str) -> dict:
    wid     = window["id"]
    tickers = data["tickers"]

    # Slice training data to window
    wdata = loader.get_window_data(data, window["start"], cfg.TRAIN_END)

    if wdata["n_train"] < cfg.LOOKBACK * 3:
        print(f"  Window {wid} skipped — insufficient data ({wdata['n_train']})")
        return None

    print(f"\n  Window {wid}: {window['start']} → {cfg.TRAIN_END} | "
          f"train={wdata['n_train']} | oos={wdata['n_oos']}")

    state_dim = data["n_asset_feats"] * cfg.LOOKBACK
    n_macro   = data["n_macro_feats"]

    model  = MAGAT(
        n_assets=len(tickers), state_dim=state_dim, n_macro=n_macro,
        gat_hidden=cfg.GAT_HIDDEN_DIM, gat_heads=cfg.GAT_N_HEADS,
        gat_layers=cfg.GAT_N_LAYERS, dqn_hidden=cfg.DQN_HIDDEN_DIM,
        meta_hidden=cfg.META_HIDDEN_DIM, dropout=cfg.GAT_DROPOUT,
    ).to(DEVICE)

    target    = copy.deepcopy(model)
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR_AGENT)
    buffer    = ReplayBuffer(cfg.REPLAY_BUFFER)
    epsilon   = cfg.EPSILON_START

    model_path = os.path.join(cfg.MODELS_DIR,
                              f"magat_option{option}_w{wid}.pt")
    best_oos   = -float("inf")

    for ep in range(1, cfg.N_EPISODES + 1):
        env   = TradingEnv(
            wdata["returns"], wdata["ohlcv_feat"],
            wdata["macro_feat"], tickers, mode="train"
        )
        state = env.reset()

        while not env.done:
            as_t = torch.tensor(state[0])
            mc_t = torch.tensor(state[1])

            if np.random.random() < epsilon:
                action = np.random.randint(len(tickers))
            else:
                with torch.no_grad():
                    out = model(as_t, mc_t)
                action = out["pick"]

            next_state, reward, done, _ = env.step(action)

            with torch.no_grad():
                node_emb = model.gat_encoder(as_t).numpy()
            buffer.push(
                (node_emb, action), action, reward,
                (node_emb, state[1]), done
            )
            update_agents(model, target, buffer, optimizer)
            state = next_state

        epsilon = max(cfg.EPSILON_END, epsilon * cfg.EPSILON_DECAY)

    # OOS evaluation
    oos_metrics = evaluate(
        model, wdata["oos_ret"], wdata["oos_ohlcv"],
        wdata["oos_macro"], tickers
    )
    oos_ret = oos_metrics.get("ann_return", 0)

    if oos_ret > best_oos:
        best_oos = oos_ret
        torch.save(model.state_dict(), model_path)

    print(f"  Window {wid} OOS: ann_ret={oos_ret*100:.2f}% | "
          f"sharpe={oos_metrics.get('sharpe',0):.3f} | "
          f"trades={oos_metrics.get('n_trades',0)}")

    return {
        "window_id":     wid,
        "train_start":   window["start"],
        "train_end":     cfg.TRAIN_END,
        "oos_ann_return":round(oos_metrics.get("ann_return", 0), 4),
        "oos_ann_vol":   round(oos_metrics.get("ann_vol", 0), 4),
        "oos_sharpe":    round(oos_metrics.get("sharpe", 0), 4),
        "oos_hit_rate":  round(oos_metrics.get("hit_rate", 0), 4),
        "oos_max_dd":    0.0,
        "oos_n_trades":  oos_metrics.get("n_trades", 0),
        "model_path":    model_path,
    }


def train_windows_option(option: str) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"MAGAT Shrinking Windows — "
          f"Option {'A (FI)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    master  = loader.load_master()
    data    = loader.get_option_data(option, master)

    all_results = []
    best_result = None
    best_return = -float("inf")

    for window in cfg.WINDOWS:
        result = train_window(window, data, option)
        if result is None:
            continue
        all_results.append({k: v for k, v in result.items()
                             if k != "model_path"})
        if result["oos_ann_return"] > best_return:
            best_return = result["oos_ann_return"]
            best_result = result

    if best_result is None:
        raise RuntimeError("All windows failed.")

    canonical = os.path.join(cfg.MODELS_DIR,
                             f"magat_option{option}_window_best.pt")
    shutil.copy2(best_result["model_path"], canonical)

    summary = {
        "option":              option,
        "trained_at":          datetime.utcnow().isoformat(),
        "elapsed_sec":         round(time.time() - t0, 1),
        "winning_window":      best_result["window_id"],
        "winning_train_start": best_result["train_start"],
        "winning_train_end":   best_result["train_end"],
        "oos_ann_return":      best_result["oos_ann_return"],
        "oos_ann_vol":         best_result["oos_ann_vol"],
        "oos_sharpe":          best_result["oos_sharpe"],
        "oos_hit_rate":        best_result["oos_hit_rate"],
        "oos_n_trades":        best_result["oos_n_trades"],
        "n_assets":            len(data["tickers"]),
        "tickers":             data["tickers"],
        "n_asset_feats":       data["n_asset_feats"],
        "n_macro_feats":       data["n_macro_feats"],
        "trading_cost_bps":    cfg.TRADING_COST_BPS,
        "all_windows":         all_results,
        "config": {
            "lookback":    cfg.LOOKBACK,
            "gat_hidden":  cfg.GAT_HIDDEN_DIM,
            "gat_heads":   cfg.GAT_N_HEADS,
            "gat_layers":  cfg.GAT_N_LAYERS,
            "dqn_hidden":  cfg.DQN_HIDDEN_DIM,
            "n_episodes":  cfg.N_EPISODES,
            "gamma":       cfg.GAMMA,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR,
                             f"meta_option{option}_window.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Winner: Window {best_result['window_id']} "
          f"({best_result['train_start']}→{best_result['train_end']}) "
          f"| OOS={best_result['oos_ann_return']*100:.2f}%")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]
    for opt in options:
        train_windows_option(opt)
