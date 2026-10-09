"""Macro/meso controllers for A²OPD trust-region constrained OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class A2OPDController:
    """Dual-layer controller: outer budget B_k, inner Lagrange multiplier lambda_k."""

    budget_init_mode: str = "proxy_kl"
    budget_init_manual: float = 0.1
    budget_min: float = 1e-4
    budget_max: float = 10.0
    budget_eta: float = 0.05
    budget_beta1: float = 0.9
    budget_beta2: float = 0.99
    budget_epsilon: float = 1e-8
    budget_adaptive_enable: bool = True
    budget_adaptive_source: str = "proxy_kl"
    budget_transfer_ema_tau: float = 0.05

    lambda_init: float = 1.0
    lambda_min: float = 0.01
    lambda_max: float = 10.0
    lambda_eta: float = 0.1
    lambda_dual_enable: bool = True

    budget: float | None = None
    lambda_k: float = 1.0
    budget_initialized: bool = False
    proxy_kl_ema: float | None = None

    budget_m: float = 0.0
    budget_v: float = 0.0
    budget_prev_reward: float | None = None

    def __post_init__(self) -> None:
        self.lambda_k = float(self.lambda_init)
        if self.is_budget_fixed() and self.budget_init_mode == "manual":
            self.budget = float(np.clip(self.budget_init_manual, self.budget_min, self.budget_max))
            self.budget_initialized = True

    @classmethod
    def from_config(cls, cfg: Any) -> "A2OPDController":
        source = str(cfg.get("budget_adaptive_source", "proxy_kl"))
        enabled = bool(cfg.get("budget_adaptive_enable", True))
        if enabled and source not in {"proxy_kl", "target_transfer_kl", "val_reward"}:
            raise ValueError(
                "a2opd.budget_adaptive_source must be 'proxy_kl', "
                "'target_transfer_kl', or 'val_reward'."
            )
        if source == "val_reward" and not enabled:
            raise ValueError("a2opd.budget_adaptive_source='val_reward' requires budget_adaptive_enable=true.")
        return cls(
            budget_init_mode=str(cfg.get("budget_init_mode", "proxy_kl")),
            budget_init_manual=float(cfg.get("budget_init_manual", 0.1)),
            budget_min=float(cfg.get("budget_min", 1e-4)),
            budget_max=float(cfg.get("budget_max", 10.0)),
            budget_eta=float(cfg.get("budget_eta", 0.05)),
            budget_beta1=float(cfg.get("budget_beta1", 0.9)),
            budget_beta2=float(cfg.get("budget_beta2", 0.99)),
            budget_epsilon=float(cfg.get("budget_epsilon", 1e-8)),
            budget_adaptive_enable=enabled,
            budget_adaptive_source=source,
            budget_transfer_ema_tau=float(cfg.get("budget_transfer_ema_tau", 0.05)),
            lambda_init=float(cfg.get("lambda_init", 1.0)),
            lambda_min=float(cfg.get("lambda_min", 0.01)),
            lambda_max=float(cfg.get("lambda_max", 10.0)),
            lambda_eta=float(cfg.get("lambda_eta", 0.1)),
            lambda_dual_enable=bool(cfg.get("lambda_dual_enable", True)),
        )

    def is_budget_fixed(self) -> bool:
        return not self.budget_adaptive_enable

    def is_budget_proxy_adaptive(self) -> bool:
        return self.budget_adaptive_enable and self.budget_adaptive_source == "proxy_kl"

    def is_budget_target_transfer_adaptive(self) -> bool:
        return self.budget_adaptive_enable and self.budget_adaptive_source == "target_transfer_kl"

    def is_budget_val_adaptive(self) -> bool:
        return self.budget_adaptive_enable and self.budget_adaptive_source == "val_reward"

    def maybe_init_budget_from_proxy_kl(self, proxy_kl: float) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.budget_initialized:
            return metrics
        if self.budget_init_mode != "proxy_kl":
            return metrics
        self.budget = float(np.clip(max(proxy_kl, 0.0), self.budget_min, self.budget_max))
        self.budget_initialized = True
        metrics["a2opd/budget"] = self.budget
        return metrics

    def update_proxy_kl_ema(self, proxy_kl: float) -> float:
        """Track EMA of per-step proxy KL for monitoring (independent of budget source)."""
        proxy_kl = float(np.clip(max(float(proxy_kl), 0.0), self.budget_min, self.budget_max))
        if self.proxy_kl_ema is None:
            self.proxy_kl_ema = proxy_kl
        else:
            tau = self.budget_transfer_ema_tau
            self.proxy_kl_ema = float(
                np.clip(
                    (1.0 - tau) * self.proxy_kl_ema + tau * proxy_kl,
                    self.budget_min,
                    self.budget_max,
                )
            )
        return self.proxy_kl_ema

    def update_budget_from_proxy_kl(self, proxy_kl: float) -> dict[str, float]:
        """Adaptive macro budget: B_k follows current-batch teacher proxy displacement."""
        self.budget = float(np.clip(max(float(proxy_kl), 0.0), self.budget_min, self.budget_max))
        self.budget_initialized = True
        return {
            "a2opd/budget": self.budget,
        }

    def update_budget_from_target_transfer_kl(self, transfer_kl: float) -> dict[str, float]:
        """EMA update B_k from the target-space full-transfer KL estimate."""
        transfer_kl = max(float(transfer_kl), 0.0)
        transfer_kl = float(np.clip(transfer_kl, self.budget_min, self.budget_max))
        if self.budget is None or not self.budget_initialized:
            self.budget = transfer_kl
            self.budget_initialized = True
        else:
            self.budget = float(
                np.clip(
                    (1.0 - self.budget_transfer_ema_tau) * self.budget
                    + self.budget_transfer_ema_tau * transfer_kl,
                    self.budget_min,
                    self.budget_max,
                )
            )
        return {
            "a2opd/target_transfer_kl": transfer_kl,
            "a2opd/budget": self.budget,
        }

    def maybe_update_budget(self, reward: float, step: int, update_interval: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not self.is_budget_val_adaptive() or not self.budget_initialized or self.budget is None:
            return metrics
        if step % max(int(update_interval), 1) != 0:
            return metrics

        reward = float(reward)
        metrics["a2opd/budget_reward"] = reward
        if self.budget_prev_reward is None:
            self.budget_prev_reward = reward
            return metrics

        delta = (reward - self.budget_prev_reward) / (abs(self.budget_prev_reward) + self.budget_epsilon)
        self.budget_m = self.budget_beta1 * self.budget_m + (1.0 - self.budget_beta1) * delta
        self.budget_v = self.budget_beta2 * self.budget_v + (1.0 - self.budget_beta2) * delta * delta
        snr = self.budget_m / (np.sqrt(self.budget_v) + self.budget_epsilon)
        exponent = float(self.budget_eta) * snr
        self.budget = float(np.clip(self.budget * np.exp(exponent), self.budget_min, self.budget_max))
        self.budget_prev_reward = reward
        metrics.update(
            {
                "a2opd/budget_reward_delta": delta,
                "a2opd/budget_reward_snr": snr,
                "a2opd/budget": self.budget,
            }
        )
        return metrics

    def update_lambda(self, anchor_kl: float) -> dict[str, float]:
        metrics: dict[str, float] = {
            "a2opd/lambda": self.lambda_k,
            "a2opd/budget": float(self.budget if self.budget is not None else 0.0),
        }
        if not self.lambda_dual_enable or not self.budget_initialized or self.budget is None:
            return metrics

        ratio = float(anchor_kl) / max(float(self.budget), self.budget_epsilon)
        exponent = self.lambda_eta * (ratio - 1.0)
        self.lambda_k = float(np.clip(self.lambda_k * np.exp(exponent), self.lambda_min, self.lambda_max))
        metrics.update(
            {
                "a2opd/anchor_kl_ratio": ratio,
                "a2opd/lambda": self.lambda_k,
            }
        )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "lambda_k": self.lambda_k,
            "budget_initialized": self.budget_initialized,
            "proxy_kl_ema": self.proxy_kl_ema,
            "budget_m": self.budget_m,
            "budget_v": self.budget_v,
            "budget_prev_reward": self.budget_prev_reward,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        if state.get("budget") is not None:
            self.budget = float(state["budget"])
        self.lambda_k = float(state.get("lambda_k", self.lambda_k))
        self.budget_initialized = bool(state.get("budget_initialized", self.budget_initialized))
        proxy_kl_ema = state.get("proxy_kl_ema", self.proxy_kl_ema)
        self.proxy_kl_ema = None if proxy_kl_ema is None else float(proxy_kl_ema)
        self.budget_m = float(state.get("budget_m", self.budget_m))
        self.budget_v = float(state.get("budget_v", self.budget_v))
        prev_reward = state.get("budget_prev_reward", self.budget_prev_reward)
        self.budget_prev_reward = None if prev_reward is None else float(prev_reward)
