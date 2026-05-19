from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from common.actions import ACTION_NAMES, pygame_action
from common.display import obs_to_display
from core.rl import (
    ROOT,
    DoomQNetwork,
    first_obs,
    make_env,
    model_input_channels,
    save_checkpoint,
    set_seed,
)


def save_demo_episode(
    out_dir: Path,
    episode_idx: int,
    observations: list[torch.Tensor],
    actions: list[int],
    rewards: list[float],
    dones: list[bool],
    config: dict,
) -> Path | None:
    if not observations:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    out = out_dir / f"demo_{timestamp}_ep{episode_idx:03d}.pt"
    torch.save(
        {
            "obs": torch.stack(observations).to(torch.float16),
            "actions": torch.as_tensor(actions, dtype=torch.long),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32),
            "dones": torch.as_tensor(dones, dtype=torch.bool),
            "config": {
                "screen_format": config["screen_format"],
                "n_stack_frames": config["n_stack_frames"],
                "extra_state": config["extra_state"],
                "hud": config["hud"],
                "crosshair": config["crosshair"],
            },
            "action_names": ACTION_NAMES,
        },
        out,
    )
    print(f"Saved demo: {out} ({len(actions)} steps)")
    return out


def collect_demos(args, config: dict) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Install pygame to collect demos.") from exc

    set_seed(config["seed"])
    out_dir = Path(args.demo_dir).resolve()
    os.chdir(ROOT)

    pygame.init()
    pygame.display.set_caption("DOOM human demo collector")
    window = pygame.display.set_mode((args.window_size, args.window_size))
    clock = pygame.time.Clock()

    print(
        "Controls: W/UP forward, SPACE/mouse attack, A/D strafe, "
        "LEFT/RIGHT turn, J jump, ESC quit."
    )
    stop = False
    for episode_idx in range(1, args.episodes + 1):
        if stop:
            break
        env = make_env(config, config["seed"] + episode_idx, train=True)
        obs = first_obs(env.reset())
        observations: list[torch.Tensor] = []
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        episode_reward = 0.0
        done = False

        with tqdm(
            total=config["train_episode_timeout"],
            desc=f"demo episode {episode_idx}/{args.episodes}",
            unit="step",
        ) as pbar:
            while not done and not stop:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop = True
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        stop = True

                frame = obs_to_display(obs)
                surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                surface = pygame.transform.scale(
                    surface, (args.window_size, args.window_size)
                )
                window.blit(surface, (0, 0))
                pygame.display.flip()

                action = pygame_action(pygame)
                next_obs_raw, reward, done, _ = env.step(action)
                observations.append(obs.detach().cpu())
                actions.append(action)
                rewards.append(float(reward[0]))
                dones.append(bool(done))
                episode_reward += float(reward[0])
                obs = first_obs(next_obs_raw)
                pbar.update(1)
                pbar.set_postfix(action=ACTION_NAMES[action], reward=f"{episode_reward:.1f}")
                clock.tick(args.fps)

        env.close()
        save_demo_episode(out_dir, episode_idx, observations, actions, rewards, dones, config)
        print(f"Episode {episode_idx}: reward={episode_reward:.2f}, steps={len(actions)}")

    pygame.quit()


class DemoDataset(Dataset):
    def __init__(self, demo_dir: Path, keep_noop: bool = True):
        self.paths = sorted(demo_dir.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No demo .pt files found in {demo_dir}")

        self.items: list[tuple[int, int]] = []
        self.lengths: list[int] = []
        for path_idx, path in enumerate(self.paths):
            data = torch.load(path, map_location="cpu", weights_only=False)
            actions = data["actions"].long()
            self.lengths.append(int(actions.numel()))
            for local_idx, action in enumerate(actions.tolist()):
                if keep_noop or action != 0:
                    self.items.append((path_idx, local_idx))
        if not self.items:
            raise ValueError("Demo dataset has no usable actions after filtering no-op actions.")

        self._cache_path_idx = -1
        self._cache = None
        print(
            f"Loaded demo index: {len(self.paths)} file(s), "
            f"{sum(self.lengths)} raw step(s), {len(self.items)} training sample(s)."
        )

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path_idx: int) -> dict:
        if self._cache_path_idx != path_idx:
            self._cache = torch.load(
                self.paths[path_idx], map_location="cpu", weights_only=False
            )
            self._cache_path_idx = path_idx
        return self._cache

    def __getitem__(self, index: int):
        path_idx, local_idx = self.items[index]
        data = self._load(path_idx)
        obs = data["obs"][local_idx].to(torch.float32)
        action = data["actions"][local_idx].long()
        return obs, action


def demo_file_summary(demo_dir: Path, keep_noop: bool = True) -> tuple[list[Path], int, int]:
    paths = sorted(demo_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No demo .pt files found in {demo_dir}")

    raw_steps = 0
    train_steps = 0
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        actions = data["actions"].long()
        raw_steps += int(actions.numel())
        train_steps += int(actions.numel() if keep_noop else (actions != 0).sum().item())

    if train_steps == 0:
        raise ValueError("Demo dataset has no usable actions after filtering no-op actions.")
    return paths, raw_steps, train_steps


def demo_action_counts(paths: list[Path], keep_noop: bool, n_actions: int = 8) -> torch.Tensor:
    counts = torch.zeros(n_actions, dtype=torch.long)
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        actions = data["actions"].long()
        if not keep_noop:
            actions = actions[actions != 0]
        counts += torch.bincount(actions, minlength=n_actions)[:n_actions]
    return counts


def iter_demo_file_batches(
    paths: list[Path],
    batch_size: int,
    keep_noop: bool,
    *,
    shuffle_files: bool = True,
    balanced_actions: bool = False,
):
    file_order = torch.randperm(len(paths)).tolist() if shuffle_files else list(range(len(paths)))
    for path_idx in file_order:
        data = torch.load(paths[path_idx], map_location="cpu", weights_only=False)
        obs = data["obs"]
        actions = data["actions"].long()
        indices = torch.arange(actions.numel()) if keep_noop else torch.nonzero(actions != 0).flatten()
        if indices.numel() == 0:
            continue

        if not balanced_actions:
            indices = indices[torch.randperm(indices.numel())]
            for start in range(0, indices.numel(), batch_size):
                batch_idx = indices[start : start + batch_size]
                yield obs[batch_idx].to(torch.float32), actions[batch_idx]
            continue

        action_ids = sorted(set(actions[indices].tolist()))
        by_action = {
            action_id: indices[actions[indices] == action_id]
            for action_id in action_ids
        }
        batches_for_file = int(np.ceil(indices.numel() / max(batch_size, 1)))
        per_action = max(1, batch_size // max(len(action_ids), 1))
        for _ in range(batches_for_file):
            batch_parts = []
            for action_id in action_ids:
                action_indices = by_action[action_id]
                picks = torch.randint(0, action_indices.numel(), (per_action,))
                batch_parts.append(action_indices[picks])
            batch_idx = torch.cat(batch_parts)
            if batch_idx.numel() < batch_size:
                picks = torch.randint(0, indices.numel(), (batch_size - batch_idx.numel(),))
                batch_idx = torch.cat([batch_idx, indices[picks]])
            batch_idx = batch_idx[torch.randperm(batch_idx.numel())[:batch_size]]
            yield obs[batch_idx].to(torch.float32), actions[batch_idx]


def pretrain_dqn_from_demos(args, config: dict, run_dir: Path) -> Path:
    demo_dir = Path(args.demo_dir).resolve()
    keep_noop = not bool(getattr(args, "drop_noop", False))
    balance_actions = bool(getattr(args, "balance_actions", False))
    class_weight_cap = float(getattr(args, "bc_class_weight_cap", 0.0) or 0.0)
    paths, raw_steps, train_steps = demo_file_summary(
        demo_dir,
        keep_noop=keep_noop,
    )
    action_counts = demo_action_counts(paths, keep_noop=keep_noop)
    present = action_counts > 0
    class_weights = None
    if class_weight_cap > 0.0:
        class_weights = torch.zeros_like(action_counts, dtype=torch.float32)
        class_weights[present] = train_steps / (
            int(present.sum().item()) * action_counts[present].float()
        )
        class_weights = torch.clamp(class_weights, max=class_weight_cap)
    num_batches = int(np.ceil(train_steps / max(args.bc_batch_size, 1)))
    print(
        f"Loaded demo index: {len(paths)} file(s), "
        f"{raw_steps} raw step(s), {train_steps} training sample(s)."
    )
    print(
        "BC sampling: "
        f"{'keeping' if keep_noop else 'dropping'} no-op actions, "
        f"{'balanced' if balance_actions else 'natural'} action batches, "
        f"{'class weights cap=' + str(class_weight_cap) if class_weights is not None else 'no class weights'}."
    )
    print(
        "Action counts: "
        + ", ".join(
            f"{ACTION_NAMES.get(i, str(i))}={int(count)}"
            for i, count in enumerate(action_counts.tolist())
        )
    )
    if train_steps > 0:
        forward_ratio = float(action_counts[1].item()) / float(train_steps)
        turn_ratio = float((action_counts[5] + action_counts[6]).item()) / float(train_steps)
        attack_ratio = float(action_counts[2].item()) / float(train_steps)
        if forward_ratio > 0.45 and turn_ratio < 0.20:
            print(
                "WARNING: demos are dominated by move_forward and have few turns. "
                "BC may learn to run into walls; record fresh demos with explicit "
                "LEFT/RIGHT turning if this happens."
            )
        if attack_ratio < 0.05:
            print(
                "WARNING: demos contain very few attack actions. "
                "BC may learn movement but not fighting."
            )

    dqn_config = config.copy()
    dqn_config.update(
        {
            "algo": "dqn",
            "algo_type": "POLICY",
            "bc_policy": True,
            "learning_rate": config["dqn_learning_rate"],
        }
    )
    in_ch = model_input_channels(dqn_config)
    model = DoomQNetwork(in_ch).to(args.device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=dqn_config["learning_rate"], eps=1e-5)
    if class_weights is not None:
        class_weights = class_weights.to(args.device_obj)

    model.train()
    for epoch in range(1, args.bc_epochs + 1):
        losses = []
        correct = 0
        total = 0
        batches = iter_demo_file_batches(
            paths,
            args.bc_batch_size,
            keep_noop=keep_noop,
            shuffle_files=True,
            balanced_actions=balance_actions,
        )
        pbar = tqdm(
            batches,
            total=num_batches,
            desc=f"BC epoch {epoch}/{args.bc_epochs}",
            unit="batch",
        )
        for obs, actions in pbar:
            obs = obs.to(args.device_obj)
            actions = actions.to(args.device_obj)
            logits = model(obs)
            loss = F.cross_entropy(logits, actions, weight=class_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), dqn_config["max_grad_norm"])
            optimizer.step()

            losses.append(float(loss.item()))
            pred = logits.argmax(1)
            correct += int((pred == actions).sum().item())
            total += int(actions.numel())
            pbar.set_postfix(
                loss=f"{float(np.mean(losses)):.4f}",
                acc=f"{correct / max(total, 1):.3f}",
            )

        acc = correct / max(total, 1)
        print(
            f"BC epoch {epoch}/{args.bc_epochs}: "
            f"loss={float(np.mean(losses)):.4f} acc={acc:.3f}"
        )

    ckpt_path = run_dir / "ckpts" / "bc-dqn-init.pth"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        dqn_config,
        global_step=0,
        best_eval=-np.inf,
    )
    print(f"Behavior-cloned DQN checkpoint: {ckpt_path}")
    return ckpt_path
