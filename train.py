# train.py — MAGAT fixed split RL training
#
# Training loop:
#   1. Initialise MAGAT + target network + replay buffer
#   2. Run N_EPISODES through training data
#   3. Each step: MAGAT selects ETF → env returns reward with 15bps cost
#   4. Store transition → sample batch → update agent networks
#   5. Validate on val set → save best model
#   6. Evaluate on test set
#
# Usage:
#   python train.py --option A
#   python train.py --option both

import argparse
import json
import os
import pickle
import shutil
import time
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
import loader
from model import MAGAT, soft_update, count_parameters
from environment import TradingEnv, ReplayBuffer

os.makedirs(cfg.MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cpu")


# ── DQN update ────────────────────────────────────────────────────────────────

def update_agents(
    model: MAGAT,
    target: MAGAT,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
) -> float:
    if len(buffer) < cfg.MIN_REPLAY:
        return 0.0

    batch     = buffer.sample(cfg.BATCH_SIZE_RL)
    total_loss = 0.0

    for i, agent in enumerate(model.agents):
        # Collect transitions where this agent was selected
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for (as_, a_), r, (ns_, nm_), d in [
            ((b[0][0], b[0][1]), b[1], (b[3][0], b[3][1]), b[4])
            for b in batch
        ]:
            if a_ == i:
                states.append(as_[i])
                actions.append(1)       # agent was active (action=hold)
                rewards.append(r)
                next_states.append(ns_[i])
                dones.append(d)

        if not states:
            continue

        s_t  = torch.tensor(np.array(states),      dtype=torch.float32)
        r_t  = torch.tensor(np.array(rewards),     dtype=torch.float32)
        ns_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        d_t  = torch.tensor(np.array(dones),       dtype=torch.float32)

        # Current Q
        q_cur = agent(s_t)[:, 1]           # Q(active) for selected agent

        # Target Q (double DQN style)
        with torch.no_grad():
            q_next = target.agents[i](ns_t).max(dim=1)[0]
        q_target = r_t + cfg.GAMMA * q_next * (1 - d_t)

        loss = F.smooth_l1_loss(q_cur, q_target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

    soft_update(target, model, cfg.TAU)
    return total_loss / max(len(model.agents), 1)


# ── Evaluate on data split ────────────────────────────────────────────────────

def evaluate(model: MAGAT, ret: np.ndarray, ohlcv: np.ndarray,
             macro: np.ndarray, tickers: list) -> dict:
    env   = TradingEnv(ret, ohlcv, macro, tickers, mode="eval")
    state = env.reset()
    model.eval()

    picks = []
    with torch.no_grad():
        while not env.done:
            as_t = torch.tensor(state[0])
            mc_t = torch.tensor(state[1])
            out  = model(as_t, mc_t)
            action = out["pick"]
            state, _, done, _ = env.step(action)
            picks.append(action)
            if done:
                break

    metrics = env.episode_metrics()
    model.train()
    return metrics


# ── Train one option ──────────────────────────────────────────────────────────

def train_option(option: str) -> dict:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"MAGAT Fixed Split — Option {'A (FI)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    master = loader.load_master()
    data   = loader.get_option_data(option, master)
    tickers = data["tickers"]

    state_dim = data["n_asset_feats"] * cfg.LOOKBACK
    n_macro   = data["n_macro_feats"]

    model = MAGAT(
        n_assets=len(tickers),
        state_dim=state_dim,
        n_macro=n_macro,
        gat_hidden=cfg.GAT_HIDDEN_DIM,
        gat_heads=cfg.GAT_N_HEADS,
        gat_layers=cfg.GAT_N_LAYERS,
        dqn_hidden=cfg.DQN_HIDDEN_DIM,
        meta_hidden=cfg.META_HIDDEN_DIM,
        dropout=cfg.GAT_DROPOUT,
    ).to(DEVICE)

    target = copy.deepcopy(model)
    target.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR_AGENT)
    buffer    = ReplayBuffer(cfg.REPLAY_BUFFER)
    epsilon   = cfg.EPSILON_START

    model_path = os.path.join(cfg.MODELS_DIR, f"magat_option{option}_best.pt")
    best_val   = -float("inf")
    best_metrics = {}

    print(f"\n  Training {cfg.N_EPISODES} episodes | "
          f"trading_cost={cfg.TRADING_COST_BPS}bps | "
          f"state_dim={state_dim} | n_macro={n_macro}")

    for ep in range(1, cfg.N_EPISODES + 1):
        env   = TradingEnv(
            data["train_ret"], data["train_ohlcv"],
            data["train_macro"], tickers, mode="train"
        )
        state = env.reset()
        ep_loss = []

        while not env.done:
            as_t = torch.tensor(state[0])
            mc_t = torch.tensor(state[1])

            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(len(tickers))
            else:
                with torch.no_grad():
                    out = model(as_t, mc_t)
                action = out["pick"]

            next_state, reward, done, info = env.step(action)

            # Store transition — use GAT node embedding as state
            with torch.no_grad():
                node_emb = model.gat_encoder(as_t).numpy()
            buffer.push(
                (node_emb, action), action, reward,
                (node_emb, state[1]), done
            )

            loss = update_agents(model, target, buffer, optimizer)
            if loss > 0:
                ep_loss.append(loss)

            # Decay epsilon per step not per episode
            epsilon = max(cfg.EPSILON_END, epsilon * cfg.EPSILON_DECAY)
            state = next_state

        ep_metrics = env.episode_metrics()

        # Validate
        val_metrics = evaluate(
            model, data["val_ret"], data["val_ohlcv"],
            data["val_macro"], tickers
        )

        print(f"  Ep {ep:2d} | "
              f"train_ret={ep_metrics.get('ann_return',0)*100:.1f}% | "
              f"val_ret={val_metrics.get('ann_return',0)*100:.1f}% | "
              f"val_sh={val_metrics.get('sharpe',0):.2f} | "
              f"trades={ep_metrics.get('n_trades',0)} | "
              f"eps={epsilon:.3f}")

        if val_metrics.get("ann_return", -999) > best_val:
            best_val     = val_metrics.get("ann_return", -999)
            best_metrics = val_metrics
            torch.save(model.state_dict(), model_path)

    # Test evaluation
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_metrics = evaluate(
        model, data["test_ret"], data["test_ohlcv"],
        data["test_macro"], tickers
    )

    print(f"\n  Test: ann_ret={test_metrics.get('ann_return',0)*100:.2f}% | "
          f"sharpe={test_metrics.get('sharpe',0):.3f} | "
          f"trades={test_metrics.get('n_trades',0)}")

    n_params = count_parameters(model)
    elapsed  = round(time.time() - t0, 1)

    summary = {
        "option":          option,
        "trained_at":      datetime.utcnow().isoformat(),
        "elapsed_sec":     elapsed,
        "test_ann_return": test_metrics.get("ann_return", 0),
        "test_ann_vol":    test_metrics.get("ann_vol", 0),
        "test_sharpe":     test_metrics.get("sharpe", 0),
        "test_hit_rate":   test_metrics.get("hit_rate", 0),
        "test_n_trades":   test_metrics.get("n_trades", 0),
        "test_start":      data["test_start"],
        "n_params":        n_params,
        "n_assets":        len(tickers),
        "tickers":         tickers,
        "n_asset_feats":   data["n_asset_feats"],
        "n_macro_feats":   data["n_macro_feats"],
        "trading_cost_bps":cfg.TRADING_COST_BPS,
        "config": {
            "lookback":       cfg.LOOKBACK,
            "gat_hidden":     cfg.GAT_HIDDEN_DIM,
            "gat_heads":      cfg.GAT_N_HEADS,
            "gat_layers":     cfg.GAT_N_LAYERS,
            "dqn_hidden":     cfg.DQN_HIDDEN_DIM,
            "n_episodes":     cfg.N_EPISODES,
            "gamma":          cfg.GAMMA,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done in {elapsed}s | params={n_params:,}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]
    for opt in options:
        train_option(opt)
