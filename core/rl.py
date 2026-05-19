"""
Train a compact PPO agent for the JKU DOOM final challenge.

Outputs are written to:
  runs/run_YYYY_MM_DD-HH_MM_SS/
    ckpts/
    logs/
    training_curves.png
    model.onnx

The exported ONNX contains the metadata expected by server_eval_doom.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

_SOURCE_PATH = Path(__file__).resolve()
ROOT = _SOURCE_PATH.parent if (_SOURCE_PATH.parent / "arena").exists() else _SOURCE_PATH.parents[1]
sys.path.insert(0, str(ROOT))

from arena import ObsBuffer, VizdoomMPEnv  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = {
    "algo": "ppo",
    "total_steps": 900_000,
    "rollout_steps": 2048,
    "update_epochs": 4,
    "minibatch_size": 256,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "entropy_coef": 0.02,
    "entropy_coef_final": 0.002,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
    "seed": 23,
    "checkpoint_interval": 30_000,
    "eval_episodes": 30,
    "best_metric": "mean",
    "train_episode_timeout": 2000,
    "eval_episode_timeout": 2000,
    "num_bots": 4,
    "bot_skill": 0,
    "doom_map": "ROOM",
    "screen_format": 8,  # vizdoom.ScreenFormat.GRAY8
    "n_stack_frames": 4,
    "extra_state": ["labels", "depth"],
    "hud": "minimal",
    "crosshair": True,
    "algo_type": "ARGMAX",
    "reward_shaping": True,
    "action_shaping": True,
    "movement_reward_coef": 0.003,
    "movement_reward_cap": 8.0,
    "idle_distance_threshold": 0.5,
    "idle_angle_threshold": 1.0,
    "idle_grace_steps": 12,
    "idle_penalty": 0.01,
    "attack_no_hit_grace_steps": 8,
    "attack_no_hit_penalty": 0.03,
    "aim_center_reward_coef": 0.02,
    "aim_attack_reward_coef": 0.08,
    "attack_not_centered_penalty": 0.02,
}

DQN_CONFIG = {
    **CONFIG,
    "algo": "dqn",
    "total_steps": 900_000,
    "learning_rate": 1e-4,
    "gamma": 0.95,
    "checkpoint_interval": 30_000,
    "eval_episodes": 30,
    "n_stack_frames": 1,
    "extra_state": ["depth"],
    "hud": "none",
    "algo_type": "QVALUE",
    "replay_capacity": 5000,
    "batch_size": 64,
    "train_after": 1000,
    "train_every": 4,
    "target_update_interval": 2000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 700_000,
}

HYBRID_CONFIG = {
    **CONFIG,
    "algo": "hybrid",
    "total_steps": 900_000,
    "hybrid_dqn_steps": 300_000,
    "dqn_learning_rate": 1e-4,
    "replay_capacity": 8000,
    "batch_size": 64,
    "train_after": 1000,
    "train_every": 4,
    "target_update_interval": 2000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.08,
    "epsilon_decay_steps": 200_000,
    "policy_init_scale": 0.25,
}


class ShapedDoomReward:
    """Eval-aligned training reward with small anti-collapse penalties."""

    def __init__(self, num_players: int):
        self.num_players = num_players

    def reset(self) -> None:
        pass

    def __call__(self, vizdoom_reward, game_var, game_var_old, player_id):
        _ = vizdoom_reward, player_id
        if not game_var_old:
            return (0.0,)

        hit = game_var["HITCOUNT"] - game_var_old.get("HITCOUNT", game_var["HITCOUNT"])
        taken = game_var["HITS_TAKEN"] - game_var_old.get(
            "HITS_TAKEN", game_var["HITS_TAKEN"]
        )
        frag = game_var["FRAGCOUNT"] - game_var_old.get(
            "FRAGCOUNT", game_var["FRAGCOUNT"]
        )
        death = game_var["DEATHCOUNT"] - game_var_old.get(
            "DEATHCOUNT", game_var["DEATHCOUNT"]
        )
        damage_taken = game_var["DAMAGE_TAKEN"] - game_var_old.get(
            "DAMAGE_TAKEN", game_var["DAMAGE_TAKEN"]
        )

        return (
            2.0 * hit,
            -0.1 * taken,
            100.0 * frag,
            -20.0 * death,
            -0.02 * damage_taken,
            -0.002,
        )


class Logger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.log_dir = run_dir / "logs"
        self.ckpt_dir = run_dir / "ckpts"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, list[tuple[int, float, float]]] = {}
        self.dumped: dict[str, int] = {}

    def log(self, name: str, value: float) -> None:
        rows = self.data.setdefault(name, [])
        rows.append((len(rows), time(), float(value)))
        self.dumped.setdefault(name, -1)

    def values(self, name: str) -> list[float]:
        return [row[2] for row in self.data.get(name, [])]

    def dump(self) -> None:
        for name, rows in self.data.items():
            path = self.log_dir / f"{name}.log"
            with path.open("a", encoding="utf-8") as f:
                for i, row in enumerate(rows):
                    if i > self.dumped[name]:
                        f.write(",".join(str(x) for x in row) + "\n")
                        self.dumped[name] = i


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def screen_channels(screen_format: int) -> int:
    return 1 if int(screen_format) == 8 else 3


def buffer_channels(extra_state: Iterable[str] | None) -> int:
    if not extra_state:
        return 0
    channels = 0
    for item in extra_state:
        channels += 3 if item == ObsBuffer.AUTOMAP.value else 1
    return channels


def input_shape(config: dict) -> tuple[int, ...]:
    base_channels = screen_channels(config["screen_format"]) + buffer_channels(
        config["extra_state"]
    )
    n_stack = int(config["n_stack_frames"])
    if n_stack > 1:
        return base_channels, n_stack, 128, 128
    return base_channels, 128, 128


def model_input_channels(config: dict) -> int:
    return (
        screen_channels(config["screen_format"]) + buffer_channels(config["extra_state"])
    ) * int(config["n_stack_frames"])


def make_env(config: dict, seed: int, *, train: bool) -> VizdoomMPEnv:
    extra_state = None
    if config["extra_state"]:
        extra_state = [ObsBuffer(x) for x in config["extra_state"]]

    reward_fn = ShapedDoomReward(1) if train and config["reward_shaping"] else None
    return VizdoomMPEnv(
        config_path="arena/scenarios/jku.cfg",
        num_players=1,
        num_bots=config["num_bots"],
        bot_skill=config["bot_skill"],
        doom_map=config["doom_map"],
        episode_timeout=(
            config["train_episode_timeout"] if train else config["eval_episode_timeout"]
        ),
        screen_format=config["screen_format"],
        n_stack_frames=config["n_stack_frames"],
        extra_state=extra_state,
        hud=config["hud"],
        crosshair=config["crosshair"],
        reward_fn=reward_fn,
        seed=seed,
    )


def first_obs(obs):
    return obs[0].detach().float().cpu()


class DoomActorCritic(nn.Module):
    def __init__(self, in_channels: int, n_actions: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(nn.Linear(64 * 4 * 4, 512), nn.ReLU(inplace=True))
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def _flatten_stack(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
        return x.float()

    def forward(self, x: torch.Tensor):
        x = self._flatten_stack(x)
        h = self.trunk(self.encoder(x))
        return self.policy(h), self.value(h).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class DoomQNetwork(nn.Module):
    def __init__(self, in_channels: int, n_actions: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions, bias=False),
        )

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

    def _flatten_stack(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            x = x.reshape(b, c * t, h, w)
        return x.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flatten_stack(x)
        return self.head(self.encoder(x))


def copy_compatible_weights(
    source: nn.Module,
    target: nn.Module,
    *,
    scale: float = 1.0,
) -> int:
    copied = 0
    source_layers = [
        module for module in source.modules() if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    target_layers = [
        module for module in target.modules() if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    for src, dst in zip(source_layers, target_layers):
        if src.weight.shape != dst.weight.shape:
            continue
        with torch.no_grad():
            dst.weight.copy_(src.weight * scale)
            if dst.bias is not None:
                dst.bias.zero_()
        copied += 1
    return copied


def init_actor_critic_from_q(
    q_model: DoomQNetwork,
    in_channels: int,
    device: torch.device,
    policy_scale: float,
) -> DoomActorCritic:
    model = DoomActorCritic(in_channels).to(device)
    copy_compatible_weights(q_model.encoder, model.encoder)
    copy_compatible_weights(q_model.head[:2], model.trunk)
    copy_compatible_weights(q_model.head[2], model.policy, scale=policy_scale)
    return model


class ActorExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.is_q_model = isinstance(model, DoomQNetwork)
        if self.is_q_model:
            self.encoder = model.encoder
            self.head = model.head
        else:
            self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_q_model:
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if x.ndim == 5:
                b, c, t, h, w = x.shape
                x = x.reshape(b, c * t, h, w)
            return self.head(self.encoder(x.float()))
        return self.model.logits(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.obs: list[torch.Tensor | None] = [None] * self.capacity
        self.next_obs: list[torch.Tensor | None] = [None] * self.capacity
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        self.obs[self.position] = obs.detach().cpu().to(torch.float16)
        self.next_obs[self.position] = next_obs.detach().cpu().to(torch.float16)
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = np.random.randint(0, self.size, size=batch_size)
        obs = torch.stack([self.obs[i] for i in indices]).to(device, dtype=torch.float32)
        next_obs = torch.stack([self.next_obs[i] for i in indices]).to(
            device, dtype=torch.float32
        )
        actions = torch.as_tensor(self.actions[indices], dtype=torch.long, device=device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device)
        return obs, actions, rewards, next_obs, dones


ACTION_ATTACK = 2
ENEMY_LABEL_KEYWORDS = ("player", "marine", "bot")


def _player_vars(env: VizdoomMPEnv) -> dict:
    return env.envs[0].unwrapped._game_vars


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _enemy_aim_score(env: VizdoomMPEnv) -> float:
    """Return 0..1 when an enemy label is visible near the crosshair."""
    state = env.envs[0].unwrapped._state
    if state is None or state.labels_buffer is None or not state.labels:
        return 0.0

    labels_buffer = state.labels_buffer
    height, width = labels_buffer.shape[-2:]
    center_x = (width - 1) * 0.5
    center_y = (height - 1) * 0.5
    best = 0.0

    for label in state.labels:
        name = str(getattr(label, "object_name", "")).lower()
        if not any(keyword in name for keyword in ENEMY_LABEL_KEYWORDS):
            continue
        mask = labels_buffer == int(label.value)
        pixels = int(mask.sum())
        if pixels <= 0:
            continue
        ys, xs = np.nonzero(mask)
        mean_x = float(xs.mean())
        mean_y = float(ys.mean())
        x_score = 1.0 - min(abs(mean_x - center_x) / max(center_x, 1.0), 1.0)
        y_score = 1.0 - min(abs(mean_y - center_y) / max(center_y, 1.0), 1.0)
        area_score = min(pixels / float(width * height) * 80.0, 1.0)
        score = max(0.0, x_score) * max(0.0, y_score) * (0.5 + 0.5 * area_score)
        best = max(best, score)
    return float(best)


class ActionShaper:
    """Small training-only reward nudges to avoid stationary attack collapse."""

    def __init__(self, config: dict):
        self.enabled = bool(config.get("action_shaping", True))
        self.config = config
        self.prev_vars: dict | None = None
        self.idle_steps = 0
        self.attack_no_hit_steps = 0

    def reset(self, env: VizdoomMPEnv) -> None:
        self.prev_vars = copy.deepcopy(_player_vars(env)) if self.enabled else None
        self.idle_steps = 0
        self.attack_no_hit_steps = 0

    def reward(self, action: int, env: VizdoomMPEnv, done: bool) -> float:
        if not self.enabled or self.prev_vars is None:
            return 0.0

        current = _player_vars(env)
        prev = self.prev_vars
        dx = float(current.get("POSITION_X", 0.0) - prev.get("POSITION_X", 0.0))
        dy = float(current.get("POSITION_Y", 0.0) - prev.get("POSITION_Y", 0.0))
        distance = float(np.hypot(dx, dy))
        angle_change = _angle_delta(
            float(current.get("ANGLE", 0.0)),
            float(prev.get("ANGLE", 0.0)),
        )
        hit_delta = float(current.get("HITCOUNT", 0.0) - prev.get("HITCOUNT", 0.0))

        bonus = float(self.config.get("movement_reward_coef", 0.0)) * min(
            distance, float(self.config.get("movement_reward_cap", 0.0))
        )

        if (
            distance < float(self.config.get("idle_distance_threshold", 0.0))
            and angle_change < float(self.config.get("idle_angle_threshold", 0.0))
        ):
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if action == ACTION_ATTACK and hit_delta <= 0.0:
            self.attack_no_hit_steps += 1
        else:
            self.attack_no_hit_steps = 0

        idle_excess = max(
            0, self.idle_steps - int(self.config.get("idle_grace_steps", 0))
        )
        attack_excess = max(
            0,
            self.attack_no_hit_steps
            - int(self.config.get("attack_no_hit_grace_steps", 0)),
        )
        bonus -= float(self.config.get("idle_penalty", 0.0)) * idle_excess
        bonus -= float(self.config.get("attack_no_hit_penalty", 0.0)) * attack_excess
        aim_score = _enemy_aim_score(env)
        bonus += float(self.config.get("aim_center_reward_coef", 0.0)) * aim_score
        if action == ACTION_ATTACK:
            bonus += float(self.config.get("aim_attack_reward_coef", 0.0)) * aim_score
            if aim_score < 0.25 and hit_delta <= 0.0:
                bonus -= float(self.config.get("attack_not_centered_penalty", 0.0))

        self.prev_vars = copy.deepcopy(current)
        if done:
            self.prev_vars = None
            self.idle_steps = 0
            self.attack_no_hit_steps = 0
        return bonus


@dataclass
class Rollout:
    obs: list[torch.Tensor]
    actions: list[int]
    log_probs: list[float]
    rewards: list[float]
    dones: list[float]
    values: list[float]


def empty_rollout() -> Rollout:
    return Rollout([], [], [], [], [], [])


def select_action(model: DoomActorCritic, obs: torch.Tensor, device: torch.device):
    with torch.no_grad():
        logits, value = model(obs.unsqueeze(0).to(device))
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(value.item())


def compute_gae(rollout: Rollout, next_value: float, gamma: float, gae_lambda: float):
    rewards = torch.tensor(rollout.rewards, dtype=torch.float32)
    dones = torch.tensor(rollout.dones, dtype=torch.float32)
    values = torch.tensor(rollout.values + [next_value], dtype=torch.float32)

    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_update(
    model: DoomActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    next_value: float,
    config: dict,
    device: torch.device,
    entropy_coef: float,
):
    obs = torch.stack(rollout.obs).to(device)
    actions = torch.tensor(rollout.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(rollout.log_probs, dtype=torch.float32, device=device)
    advantages, returns = compute_gae(
        rollout, next_value, config["gamma"], config["gae_lambda"]
    )
    advantages = advantages.to(device)
    returns = returns.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n = len(actions)
    indices = np.arange(n)
    stats = {"loss": [], "policy_loss": [], "value_loss": [], "entropy": [], "kl": []}

    for _ in range(config["update_epochs"]):
        np.random.shuffle(indices)
        for start in range(0, n, config["minibatch_size"]):
            mb = indices[start : start + config["minibatch_size"]]
            mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)

            logits, values = model(obs[mb_t])
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_t])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[mb_t])
            pg1 = -advantages[mb_t] * ratio
            pg2 = -advantages[mb_t] * torch.clamp(
                ratio, 1.0 - config["clip_coef"], 1.0 + config["clip_coef"]
            )
            policy_loss = torch.max(pg1, pg2).mean()
            value_loss = F.mse_loss(values, returns[mb_t])
            loss = (
                policy_loss
                + config["value_coef"] * value_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs[mb_t] - new_log_probs).mean().clamp_min(0)
            stats["loss"].append(float(loss.item()))
            stats["policy_loss"].append(float(policy_loss.item()))
            stats["value_loss"].append(float(value_loss.item()))
            stats["entropy"].append(float(entropy.item()))
            stats["kl"].append(float(approx_kl.item()))

        if stats["kl"] and np.mean(stats["kl"][-max(1, n // config["minibatch_size"]) :]) > config["target_kl"]:
            break

    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    global_step: int,
    best_eval: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": config.get("algo", "ppo"),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "global_step": global_step,
            "best_eval": best_eval,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kind = ckpt.get("kind", ckpt.get("config", {}).get("algo", "ppo"))
    config = copy.deepcopy(DQN_CONFIG if kind == "dqn" else CONFIG)
    config.update(ckpt.get("config", {}))
    in_ch = (screen_channels(config["screen_format"]) + buffer_channels(config["extra_state"])) * int(
        config["n_stack_frames"]
    )
    model = DoomQNetwork(in_ch).to(device) if config.get("algo") == "dqn" else DoomActorCritic(in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer, config, int(ckpt.get("global_step", 0)), float(
        ckpt.get("best_eval", -np.inf)
    )


def onnx_config(config: dict) -> dict:
    return {
        "screen_format": config["screen_format"],
        "n_stack_frames": config["n_stack_frames"],
        "extra_state": config["extra_state"],
        "hud": config["hud"],
        "crosshair": config["crosshair"],
        "algo_type": config["algo_type"],
    }


def export_onnx(model: DoomActorCritic, config: dict, path: Path, device: torch.device) -> None:
    model.eval()
    dummy = torch.zeros((1, *input_shape(config)), dtype=torch.float32, device=device)
    wrapper = ActorExport(model).to(device).eval()
    torch.onnx.export(
        wrapper,
        dummy,
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        input_names=["frames"],
        output_names=["logits"],
        external_data=False,
    )

    proto = onnx.load(path)
    del proto.metadata_props[:]
    meta = proto.metadata_props.add()
    meta.key = "config"
    meta.value = json.dumps(onnx_config(config))
    onnx.save(proto, path)
    print(f"ONNX saved: {path}")


def evaluate_policy(
    model: DoomActorCritic,
    config: dict,
    device: torch.device,
    episodes: int,
    seed: int,
    greedy: bool = False,
) -> dict[str, float]:
    if episodes <= 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "robust": 0.0}
    scores = []
    model.eval()
    for i in range(episodes):
        env = make_env(config, seed + 100_000 + i, train=False)
        obs = first_obs(env.reset())
        done = False
        score = 0.0
        while not done:
            with torch.no_grad():
                out = model(obs.unsqueeze(0).to(device))
                logits = out[0] if isinstance(out, tuple) else out
                use_argmax = greedy or config.get("algo_type") not in (None, "POLICY")
                action = int(logits.argmax(-1).item()) if use_argmax else int(
                    Categorical(logits=logits).sample().item()
                )
            next_obs, reward, done, _ = env.step(action)
            score += float(reward[0])
            obs = first_obs(next_obs)
        env.close()
        scores.append(score)
    mean_score = float(np.mean(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0
    min_score = float(np.min(scores)) if scores else 0.0
    max_score = float(np.max(scores)) if scores else 0.0
    robust_score = mean_score - 0.5 * std_score + 0.1 * min_score
    print(
        f"Eval over {episodes} episode(s): "
        f"mean={mean_score:.2f} std={std_score:.2f} "
        f"min={min_score:.2f} max={max_score:.2f} robust={robust_score:.2f} "
        f"scores={scores}"
    )
    return {
        "mean": mean_score,
        "std": std_score,
        "min": min_score,
        "max": max_score,
        "robust": robust_score,
    }


def plot_training_curves(logger: Logger, run_dir: Path) -> None:
    rewards = logger.values("episode_reward")
    evals = logger.values("eval_score")
    eval_mins = logger.values("eval_min")
    eval_robust = logger.values("eval_robust")
    losses = logger.values("loss")
    entropy = logger.values("entropy")
    entropy_coef = logger.values("entropy_coef")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].set_title("Episode reward")
    axes[0].plot(rewards, alpha=0.5)
    if len(rewards) >= 20:
        smooth = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        axes[0].plot(range(19, len(rewards)), smooth, label="avg20")
        axes[0].legend()

    axes[1].set_title("Eval score")
    axes[1].plot(evals, marker="o", markersize=3, label="mean")
    if eval_mins:
        axes[1].plot(eval_mins, marker=".", markersize=3, label="min")
    if eval_robust:
        axes[1].plot(eval_robust, marker=".", markersize=3, label="robust")
    axes[1].legend()

    axes[2].set_title("PPO loss")
    axes[2].plot(losses)

    axes[3].set_title("Entropy")
    axes[3].plot(entropy)
    if entropy_coef:
        ax_coef = axes[3].twinx()
        ax_coef.plot(entropy_coef, color="tab:orange", alpha=0.6, label="coef")
        ax_coef.set_ylabel("coef")

    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = run_dir / "training_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Training curves saved: {out}")


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-step(\d+)\.pth", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    best = run_dir / "ckpts" / "best-checkpoint.pth"
    if best.exists():
        return best
    ckpts = list((run_dir / "ckpts").glob("checkpoint-step*.pth"))
    return max(ckpts, key=checkpoint_step) if ckpts else None


def apply_overrides(config: dict, overrides: dict) -> dict:
    config.update({k: v for k, v in overrides.items() if v is not None})
    if config.get("no_eval"):
        config["eval_episodes"] = 0
    config.pop("no_eval", None)
    return config


def apply_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def train(
    config: dict,
    run_dir: Path,
    resume: Path | None,
    device: torch.device,
    overrides: dict | None = None,
) -> None:
    set_seed(config["seed"])
    logger = Logger(run_dir)

    if resume:
        model, optimizer, config, global_step, best_eval = load_checkpoint(resume, device)
        if overrides:
            config = apply_overrides(config, overrides)
            apply_optimizer_lr(optimizer, config["learning_rate"])
        print(f"Resumed checkpoint: {resume}")
    else:
        if overrides:
            config = apply_overrides(config, overrides)
        in_ch = (
            screen_channels(config["screen_format"]) + buffer_channels(config["extra_state"])
        ) * int(config["n_stack_frames"])
        model = DoomActorCritic(in_ch).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-5)
        global_step = 0
        best_eval = -np.inf

    best_metric_name = str(config.get("best_metric", "mean"))
    env = make_env(config, config["seed"], train=True)
    obs = first_obs(env.reset())
    action_shaper = ActionShaper(config)
    action_shaper.reset(env)
    episode_reward = 0.0
    episode_len = 0
    episodes = 0
    next_checkpoint = global_step + config["checkpoint_interval"]
    stop_requested = False

    def _handle_stop(signum, frame):
        nonlocal stop_requested
        _ = signum, frame
        stop_requested = True
        print("\nStop requested; finishing current rollout and saving.")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    pbar = tqdm(total=config["total_steps"], initial=global_step, desc="Doom PPO", unit="step")
    while global_step < config["total_steps"] and not stop_requested:
        rollout = empty_rollout()
        model.eval()

        for _ in range(config["rollout_steps"]):
            action, log_prob, value = select_action(model, obs, device)
            next_obs, reward, done, _ = env.step(action)
            train_reward = float(reward[0]) + action_shaper.reward(action, env, done)

            rollout.obs.append(obs)
            rollout.actions.append(action)
            rollout.log_probs.append(log_prob)
            rollout.rewards.append(train_reward)
            rollout.dones.append(float(done))
            rollout.values.append(value)

            episode_reward += train_reward
            episode_len += 1
            global_step += 1
            pbar.update(1)

            obs = first_obs(next_obs)
            if done:
                logger.log("episode_reward", episode_reward)
                logger.log("episode_length", episode_len)
                episodes += 1
                pbar.set_postfix(
                    ep=episodes,
                    reward=f"{episode_reward:.1f}",
                    **{f"best_{best_metric_name}": f"{best_eval:.1f}"},
                )
                episode_reward = 0.0
                episode_len = 0
                obs = first_obs(env.reset())
                action_shaper.reset(env)

            if global_step >= config["total_steps"] or stop_requested:
                break

        with torch.no_grad():
            _, next_value_t = model(obs.unsqueeze(0).to(device))
        model.train()
        progress = min(global_step / max(config["total_steps"], 1), 1.0)
        entropy_coef = float(
            config["entropy_coef"]
            + (config["entropy_coef_final"] - config["entropy_coef"]) * progress
        )
        stats = ppo_update(
            model,
            optimizer,
            rollout,
            float(next_value_t.item()),
            config,
            device,
            entropy_coef,
        )
        for name, value in stats.items():
            logger.log(name, value)
        logger.log("entropy_coef", entropy_coef)
        logger.dump()

        if global_step >= next_checkpoint or global_step >= config["total_steps"] or stop_requested:
            ckpt_path = run_dir / "ckpts" / f"checkpoint-step{global_step}.pth"
            save_checkpoint(ckpt_path, model, optimizer, config, global_step, best_eval)
            if config["eval_episodes"] > 0:
                eval_stats = evaluate_policy(
                    model,
                    config,
                    device,
                    episodes=config["eval_episodes"],
                    seed=config["seed"],
                )
                eval_score = eval_stats["mean"]
                eval_metric = eval_stats.get(best_metric_name, eval_stats["mean"])
                logger.log("eval_score", eval_score)
                logger.log("eval_std", eval_stats["std"])
                logger.log("eval_min", eval_stats["min"])
                logger.log("eval_max", eval_stats["max"])
                logger.log("eval_robust", eval_stats["robust"])
                logger.log("eval_best_metric", eval_metric)
                logger.dump()
                if eval_metric > best_eval:
                    best_eval = eval_metric
                    save_checkpoint(
                        run_dir / "ckpts" / "best-checkpoint.pth",
                        model,
                        optimizer,
                        config,
                        global_step,
                        best_eval,
                    )
            export_onnx(model, config, run_dir / "model.onnx", device)
            plot_training_curves(logger, run_dir)
            next_checkpoint += config["checkpoint_interval"]

    env.close()
    pbar.close()
    save_checkpoint(
        run_dir / "ckpts" / "final-checkpoint.pth",
        model,
        optimizer,
        config,
        global_step,
        best_eval,
    )
    best_ckpt = latest_checkpoint(run_dir)
    if best_ckpt:
        best_model, _, best_cfg, _, _ = load_checkpoint(best_ckpt, device)
        export_onnx(best_model, best_cfg, run_dir / "model.onnx", device)
    plot_training_curves(logger, run_dir)
    print(f"Run saved to: {run_dir}")


def dqn_epsilon(config: dict, global_step: int) -> float:
    frac = min(global_step / max(config["epsilon_decay_steps"], 1), 1.0)
    return float(
        config["epsilon_start"] + (config["epsilon_end"] - config["epsilon_start"]) * frac
    )


@torch.no_grad()
def select_dqn_action(
    model: DoomQNetwork,
    obs: torch.Tensor,
    config: dict,
    device: torch.device,
    global_step: int,
) -> tuple[int, float]:
    epsilon = dqn_epsilon(config, global_step)
    if random.random() < epsilon:
        return random.randrange(8), epsilon
    q_values = model(obs.unsqueeze(0).to(device))
    return int(q_values.argmax(-1).item()), epsilon


def dqn_update(
    model: DoomQNetwork,
    target_model: DoomQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    obs, actions, rewards, next_obs, dones = replay.sample(config["batch_size"], device)
    q = model(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = model(next_obs).argmax(1)
        next_q = target_model(next_obs).gather(1, next_actions[:, None]).squeeze(1)
        target = rewards + config["gamma"] * (1.0 - dones) * next_q
    loss = F.smooth_l1_loss(q, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "q_value": float(q.mean().item()),
        "target_q": float(target.mean().item()),
    }


def train_dqn(
    config: dict,
    run_dir: Path,
    resume: Path | None,
    device: torch.device,
    overrides: dict | None = None,
) -> None:
    set_seed(config["seed"])
    logger = Logger(run_dir)

    if resume:
        model, optimizer, config, global_step, best_eval = load_checkpoint(resume, device)
        if config.get("algo") != "dqn":
            raise ValueError("Cannot resume DQN training from a non-DQN checkpoint.")
        if overrides:
            config = apply_overrides(config, overrides)
            apply_optimizer_lr(optimizer, config["learning_rate"])
        model = model.to(device)
        print(f"Resumed DQN checkpoint: {resume}")
    else:
        if overrides:
            config = apply_overrides(config, overrides)
        in_ch = (
            screen_channels(config["screen_format"]) + buffer_channels(config["extra_state"])
        ) * int(config["n_stack_frames"])
        model = DoomQNetwork(in_ch).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-5)
        global_step = 0
        best_eval = -np.inf

    best_metric_name = str(config.get("best_metric", "mean"))
    target_model = copy.deepcopy(model).to(device).eval()
    replay = ReplayBuffer(config["replay_capacity"])
    env = make_env(config, config["seed"], train=True)
    obs = first_obs(env.reset())
    action_shaper = ActionShaper(config)
    action_shaper.reset(env)
    episode_reward = 0.0
    episode_len = 0
    episodes = 0
    next_checkpoint = global_step + config["checkpoint_interval"]
    stop_requested = False

    def _handle_stop(signum, frame):
        nonlocal stop_requested
        _ = signum, frame
        stop_requested = True
        print("\nStop requested; saving after current step.")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    pbar = tqdm(total=config["total_steps"], initial=global_step, desc="Doom DQN", unit="step")
    while global_step < config["total_steps"] and not stop_requested:
        model.eval()
        action, epsilon = select_dqn_action(model, obs, config, device, global_step)
        next_obs_raw, reward, done, _ = env.step(action)
        next_obs = first_obs(next_obs_raw)
        train_reward = float(reward[0]) + action_shaper.reward(action, env, done)
        replay.add(obs, action, train_reward, next_obs, done)

        episode_reward += train_reward
        episode_len += 1
        obs = next_obs
        global_step += 1
        pbar.update(1)

        if done:
            logger.log("episode_reward", episode_reward)
            logger.log("episode_length", episode_len)
            episodes += 1
            pbar.set_postfix(
                ep=episodes,
                reward=f"{episode_reward:.1f}",
                eps=f"{epsilon:.3f}",
                **{f"best_{best_metric_name}": f"{best_eval:.1f}"},
            )
            episode_reward = 0.0
            episode_len = 0
            obs = first_obs(env.reset())
            action_shaper.reset(env)

        if replay.size >= config["train_after"] and global_step % config["train_every"] == 0:
            model.train()
            stats = dqn_update(model, target_model, optimizer, replay, config, device)
            for name, value in stats.items():
                logger.log(name, value)
            logger.log("epsilon", epsilon)
            logger.dump()

        if global_step % config["target_update_interval"] == 0:
            target_model.load_state_dict(model.state_dict())

        if global_step >= next_checkpoint or global_step >= config["total_steps"] or stop_requested:
            ckpt_path = run_dir / "ckpts" / f"checkpoint-step{global_step}.pth"
            save_checkpoint(ckpt_path, model, optimizer, config, global_step, best_eval)
            if config["eval_episodes"] > 0:
                eval_stats = evaluate_policy(
                    model,
                    config,
                    device,
                    episodes=config["eval_episodes"],
                    seed=config["seed"],
                    greedy=True,
                )
                eval_metric = eval_stats.get(best_metric_name, eval_stats["mean"])
                logger.log("eval_score", eval_stats["mean"])
                logger.log("eval_std", eval_stats["std"])
                logger.log("eval_min", eval_stats["min"])
                logger.log("eval_max", eval_stats["max"])
                logger.log("eval_robust", eval_stats["robust"])
                logger.log("eval_best_metric", eval_metric)
                logger.dump()
                if eval_metric > best_eval:
                    best_eval = eval_metric
                    save_checkpoint(
                        run_dir / "ckpts" / "best-checkpoint.pth",
                        model,
                        optimizer,
                        config,
                        global_step,
                        best_eval,
                    )
            export_onnx(model, config, run_dir / "model.onnx", device)
            plot_training_curves(logger, run_dir)
            next_checkpoint += config["checkpoint_interval"]

    env.close()
    pbar.close()
    save_checkpoint(
        run_dir / "ckpts" / "final-checkpoint.pth",
        model,
        optimizer,
        config,
        global_step,
        best_eval,
    )
    best_ckpt = latest_checkpoint(run_dir)
    if best_ckpt:
        best_model, _, best_cfg, _, _ = load_checkpoint(best_ckpt, device)
        export_onnx(best_model, best_cfg, run_dir / "model.onnx", device)
    plot_training_curves(logger, run_dir)
    print(f"DQN run saved to: {run_dir}")


def train_hybrid(
    config: dict,
    run_dir: Path,
    resume: Path | None,
    device: torch.device,
    overrides: dict | None = None,
) -> None:
    if resume:
        print("Continuing hybrid PPO phase from checkpoint.")
        train(config, run_dir, resume, device, overrides)
        return

    if overrides:
        config = apply_overrides(config, overrides)

    dqn_steps = min(int(config["hybrid_dqn_steps"]), int(config["total_steps"]))
    dqn_ckpt = None
    if dqn_steps > 0:
        dqn_config = copy.deepcopy(config)
        dqn_config.update(
            {
                "algo": "dqn",
                "algo_type": "QVALUE",
                "total_steps": dqn_steps,
                "learning_rate": config["dqn_learning_rate"],
            }
        )
        dqn_run_dir = run_dir / "dqn_warmup"
        dqn_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Hybrid phase 1/2: DQN warm-up for {dqn_steps} step(s).")
        train_dqn(dqn_config, dqn_run_dir, resume=None, device=device, overrides=None)
        dqn_ckpt = latest_checkpoint(dqn_run_dir)

    ppo_config = copy.deepcopy(config)
    ppo_config["algo"] = "hybrid"
    in_ch = model_input_channels(ppo_config)
    if dqn_ckpt is not None:
        q_model, _, _, dqn_global_step, _ = load_checkpoint(dqn_ckpt, device)
        if not isinstance(q_model, DoomQNetwork):
            raise TypeError(f"Expected DQN warm-up checkpoint, got {type(q_model).__name__}")
        model = init_actor_critic_from_q(
            q_model,
            in_ch,
            device,
            policy_scale=float(config["policy_init_scale"]),
        )
        global_step = dqn_global_step
        print(f"Hybrid phase 2/2: PPO fine-tune initialized from {dqn_ckpt}.")
    else:
        model = DoomActorCritic(in_ch).to(device)
        global_step = 0
        print("Hybrid phase 2/2: PPO fine-tune from a fresh policy.")

    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config["learning_rate"], eps=1e-5)
    init_ckpt = run_dir / "ckpts" / "hybrid-ppo-init.pth"
    save_checkpoint(
        init_ckpt,
        model,
        optimizer,
        ppo_config,
        global_step,
        best_eval=-np.inf,
    )
    train(ppo_config, run_dir, init_ckpt, device, overrides=None)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or export a PPO, DQN, or hybrid DOOM agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m core.rl --total-steps 5000000 --eval-episodes 20\n"
            "  python -m core.rl --algo hybrid --hybrid-dqn-steps 200000\n"
            "  python -m core.rl --resume runs/run_2026_05_16-15_12_00\n"
            "  python -m core.rl --export runs/run_.../ckpts/best-checkpoint.pth\n\n"
            "For 120+ scores, prefer long fresh runs and evaluate with utility.eval over "
            "50+ episodes. Resuming preserves model/observation settings, while "
            "scalar CLI overrides are applied after checkpoint load."
        ),
    )
    parser.add_argument(
        "--algo",
        choices=("ppo", "dqn", "hybrid"),
        default=CONFIG["algo"],
        help="Training algorithm for fresh runs. Resume uses the checkpoint kind.",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Explicit output directory for a fresh run. If omitted, creates runs/run_TIMESTAMP.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Run directory or checkpoint to continue. Directory resumes best/latest checkpoint.",
    )
    parser.add_argument(
        "--export",
        default="",
        help="Run directory or checkpoint to export to ONNX, then exit without training.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output ONNX path for --export. Defaults to RUN_DIR/model.onnx.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help=f"Total environment steps to train. CONFIG default is {CONFIG['total_steps']}.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help=f"Steps collected before each PPO update. CONFIG default is {CONFIG['rollout_steps']}.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=f"Episodes per periodic local eval. CONFIG default is {CONFIG['eval_episodes']}.",
    )
    parser.add_argument(
        "--best-metric",
        choices=("mean", "robust"),
        default=None,
        help="Eval metric used to update ckpts/best-checkpoint.pth.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip periodic eval during training. Useful for fast sweeps; model still exports.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed. CONFIG default is {CONFIG['seed']}.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer LR.")
    parser.add_argument("--entropy-coef", type=float, default=None, help="Override PPO entropy start.")
    parser.add_argument("--entropy-coef-final", type=float, default=None, help="Override PPO entropy end.")
    parser.add_argument("--clip-coef", type=float, default=None, help="Override PPO clip coefficient.")
    parser.add_argument("--target-kl", type=float, default=None, help="Override PPO target KL.")
    parser.add_argument("--epsilon-start", type=float, default=None, help="Override DQN initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Override DQN final epsilon.")
    parser.add_argument("--epsilon-decay-steps", type=int, default=None, help="Override DQN epsilon decay steps.")
    parser.add_argument(
        "--hybrid-dqn-steps",
        type=int,
        default=None,
        help=f"DQN warm-up steps before PPO in hybrid mode. Default is {HYBRID_CONFIG['hybrid_dqn_steps']}.",
    )
    parser.add_argument(
        "--dqn-learning-rate",
        type=float,
        default=None,
        help=f"DQN warm-up optimizer LR in hybrid mode. Default is {HYBRID_CONFIG['dqn_learning_rate']}.",
    )
    parser.add_argument(
        "--policy-init-scale",
        type=float,
        default=None,
        help=f"Scale for copying DQN action weights into PPO logits. Default is {HYBRID_CONFIG['policy_init_scale']}.",
    )
    parser.add_argument("--movement-reward-coef", type=float, default=None, help="Training-only reward per movement unit, capped per step.")
    parser.add_argument("--idle-penalty", type=float, default=None, help="Training-only repeated idle penalty.")
    parser.add_argument("--attack-no-hit-penalty", type=float, default=None, help="Training-only repeated attack-without-hit penalty.")
    parser.add_argument("--aim-center-reward-coef", type=float, default=None, help="Training-only reward for visible enemy labels near screen center.")
    parser.add_argument("--aim-attack-reward-coef", type=float, default=None, help="Training-only reward for attacking while an enemy is centered.")
    parser.add_argument("--attack-not-centered-penalty", type=float, default=None, help="Training-only penalty for attacking when no enemy is centered.")
    return parser.parse_args()


def build_overrides(args) -> dict:
    return {
        "total_steps": args.total_steps,
        "rollout_steps": args.rollout_steps,
        "eval_episodes": args.eval_episodes,
        "best_metric": args.best_metric,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "entropy_coef_final": args.entropy_coef_final,
        "clip_coef": args.clip_coef,
        "target_kl": args.target_kl,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "hybrid_dqn_steps": args.hybrid_dqn_steps,
        "dqn_learning_rate": args.dqn_learning_rate,
        "policy_init_scale": args.policy_init_scale,
        "movement_reward_coef": args.movement_reward_coef,
        "idle_penalty": args.idle_penalty,
        "attack_no_hit_penalty": args.attack_no_hit_penalty,
        "aim_center_reward_coef": args.aim_center_reward_coef,
        "aim_attack_reward_coef": args.aim_attack_reward_coef,
        "attack_not_centered_penalty": args.attack_not_centered_penalty,
        "no_eval": args.no_eval,
    }


def resolve_resume(path: str) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        return latest_checkpoint(p.resolve())
    return p.resolve()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    base_config = {"ppo": CONFIG, "dqn": DQN_CONFIG, "hybrid": HYBRID_CONFIG}[args.algo]
    config = copy.deepcopy(base_config)
    overrides = build_overrides(args)
    config = apply_overrides(config, overrides)

    if args.export:
        ckpt_path = resolve_resume(args.export)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found for export: {args.export}")
        model, _, ckpt_config, _, _ = load_checkpoint(ckpt_path, device)
        out = Path(args.out).resolve() if args.out else ckpt_path.parents[1] / "model.onnx"
        os.chdir(ROOT)
        export_onnx(model, ckpt_config, out, device)
        return

    resume = resolve_resume(args.resume)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif resume:
        run_dir = resume.parents[1]
    else:
        run_dir = ROOT / "runs" / f"run_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    print(f"Device : {device}")
    print(f"Run dir: {run_dir}")
    if resume:
        ckpt_kind = torch.load(resume, map_location="cpu", weights_only=False).get("kind", "")
        if ckpt_kind == "dqn":
            train_dqn(config, run_dir, resume, device, overrides)
            return
        if ckpt_kind == "hybrid":
            train(config, run_dir, resume, device, overrides)
            return
    if config.get("algo") == "hybrid":
        train_hybrid(config, run_dir, resume, device, overrides=None)
        return
    if config.get("algo") == "dqn":
        train_dqn(config, run_dir, resume, device, overrides)
    else:
        train(config, run_dir, resume, device, overrides)


if __name__ == "__main__":
    main()
