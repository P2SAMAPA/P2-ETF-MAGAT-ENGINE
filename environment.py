# environment.py — RL Trading Environment for MAGAT
#
# The environment simulates daily ETF trading with:
# - Reward = log return of held ETF - trading cost if switched
# - Trading cost = 15 bps (0.0015) applied when ETF selection changes
# - Episode = sequential walk through training data
# - State = LOOKBACK-day window of asset features + current macro

import numpy as np
import config as cfg


class TradingEnv:
    """
    Multi-asset daily trading environment.

    State:
        asset_states: (n_assets, state_dim) — flattened LOOKBACK window per asset
        macro_state:  (n_macro,)            — current macro values

    Action:
        int — index of ETF to hold (0 to n_assets-1)
        Selected by MAGAT as the active agent with highest meta score.

    Reward:
        log_return(selected_ETF) - trading_cost * I(ETF changed)

    The 15 bps cost is applied when the held ETF changes.
    This discourages daily flipping and rewards conviction.
    """

    def __init__(
        self,
        returns: np.ndarray,          # (T, n_assets) log returns
        ohlcv_feat: np.ndarray,       # (T, n_assets, n_asset_feats) per-asset features
        macro_feat: np.ndarray,       # (T, n_macro) macro features
        tickers: list,
        lookback: int = cfg.LOOKBACK,
        trading_cost: float = cfg.TRADING_COST,
        mode: str = "train",          # "train" or "eval"
    ):
        self.returns      = returns
        self.ohlcv_feat   = ohlcv_feat
        self.macro_feat   = macro_feat
        self.tickers      = tickers
        self.n_assets     = len(tickers)
        self.n_steps      = len(returns)
        self.lookback     = lookback
        self.trading_cost = trading_cost
        self.mode         = mode

        self.state_dim = ohlcv_feat.shape[-1] * lookback  # flattened per asset
        self.n_macro   = macro_feat.shape[-1]

        self.reset()

    def reset(self) -> tuple:
        """Reset environment to start. Returns (asset_states, macro_state)."""
        self.t           = self.lookback       # start after enough history
        self.held_etf    = None                # no position initially
        self.portfolio_value = 1.0
        self.done        = False
        self.n_trades    = 0
        self.episode_rets = []

        return self._get_state()

    def _get_state(self) -> tuple:
        """Get current state as (asset_states, macro_state)."""
        # Per-asset: flatten LOOKBACK window of features
        window = self.ohlcv_feat[self.t - self.lookback: self.t]  # (L, A, F)
        # Reshape to (A, L*F)
        asset_states = window.transpose(1, 0, 2).reshape(
            self.n_assets, self.lookback * self.ohlcv_feat.shape[-1]
        )
        macro_state = self.macro_feat[self.t]                     # (n_macro,)
        return asset_states.astype(np.float32), macro_state.astype(np.float32)

    def step(self, action: int) -> tuple:
        """
        Take one step.

        action: int — ETF index to hold today

        Returns:
            next_state: (asset_states, macro_state)
            reward:     float
            done:       bool
            info:       dict
        """
        # Trading cost if switching
        traded = (self.held_etf is not None and action != self.held_etf)
        cost   = self.trading_cost if traded else 0.0

        # Return for today
        ret    = float(self.returns[self.t, action])

        # Reward = log return - cost
        reward = ret - cost

        # Track
        self.portfolio_value *= (1 + ret - cost)
        self.episode_rets.append(ret - cost)
        if traded:
            self.n_trades += 1

        self.held_etf = action
        self.t       += 1

        # Done if end of data
        self.done = (self.t >= self.n_steps - 1)

        next_state = self._get_state() if not self.done else (
            np.zeros((self.n_assets, self.lookback * self.ohlcv_feat.shape[-1]),
                     dtype=np.float32),
            np.zeros(self.n_macro, dtype=np.float32),
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "n_trades":        self.n_trades,
            "ret":             ret,
            "cost":            cost,
            "ticker":          self.tickers[action],
        }

        return next_state, reward, self.done, info

    def episode_metrics(self) -> dict:
        """Compute episode performance metrics."""
        if not self.episode_rets:
            return {}
        r = np.array(self.episode_rets)
        ar = float(r.mean() * 252)
        av = float(r.std() * np.sqrt(252) + 1e-8)
        return {
            "ann_return":      round(ar, 4),
            "ann_vol":         round(av, 4),
            "sharpe":          round(ar / av, 4),
            "portfolio_value": round(self.portfolio_value, 4),
            "n_trades":        self.n_trades,
            "hit_rate":        round(float((r > 0).mean()), 4),
        }


class ReplayBuffer:
    """
    Experience replay buffer shared across all agents.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = cfg.REPLAY_BUFFER):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]

    def __len__(self):
        return len(self.buffer)
