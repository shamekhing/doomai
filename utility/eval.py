"""
Local evaluator for the JKU DOOM challenge.

This mirrors the important parts of the assignment server evaluator:
  - reads ONNX metadata key "config"
  - builds ROOM with 4 easy bots
  - samples Categorical(logits) for POLICY models

It also accepts local .pth checkpoints so unfinished training runs can be
evaluated before exporting model.onnx.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx2pytorch import ConvertModel
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena import ObsBuffer, VizdoomMPEnv  # noqa: E402
from core.rl import load_checkpoint, onnx_config  # noqa: E402


DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_config(model_proto) -> dict:
    default = {
        "screen_format": 8,
        "n_stack_frames": 4,
        "extra_state": None,
        "hud": "minimal",
        "crosshair": True,
        "algo_type": "POLICY",
    }
    for prop in model_proto.metadata_props:
        if prop.key == "config":
            default.update(json.loads(prop.value))
            break
    return default


def resolve_target(target: str) -> Path | None:
    if not target:
        return latest_model()
    path = Path(target).resolve()
    if path.is_dir():
        onnx_path = path / "model.onnx"
        if onnx_path.exists():
            return onnx_path
        best = path / "ckpts" / "best-checkpoint.pth"
        if best.exists():
            return best
        final = path / "ckpts" / "final-checkpoint.pth"
        if final.exists():
            return final
        ckpts = sorted((path / "ckpts").glob("*.pth"))
        if ckpts:
            return ckpts[-1]
    return path


def load_policy(path: Path, device: torch.device):
    if path.suffix == ".onnx":
        model_proto = onnx.load(path)
        config = read_config(model_proto)
        model = ConvertModel(model_proto).to(device, dtype=DTYPE).eval()
        return model, config

    model, _, config, _, _ = load_checkpoint(path, device)
    model = model.to(device, dtype=DTYPE).eval()
    export_config = onnx_config(config)
    if config.get("bc_policy") or path.name == "bc-dqn-init.pth":
        export_config["algo_type"] = "POLICY"
    return model, export_config


def make_env(seed: int, config: dict) -> VizdoomMPEnv:
    extra_state = None
    if config["extra_state"] is not None:
        extra_state = []
        for item in config["extra_state"]:
            if item in {e.value for e in ObsBuffer}:
                extra_state.append(ObsBuffer(item))

    return VizdoomMPEnv(
        config_path="arena/scenarios/jku.cfg",
        num_players=1,
        num_bots=4,
        bot_skill=0,
        doom_map="ROOM",
        episode_timeout=2000,
        screen_format=config["screen_format"],
        n_stack_frames=config["n_stack_frames"],
        extra_state=extra_state,
        hud=config["hud"],
        crosshair=config["crosshair"],
        seed=seed,
    )


class Agent:
    def __init__(self, model: torch.nn.Module, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def select_action(self, frames: torch.Tensor, greedy: bool = False) -> int:
        frames = frames.unsqueeze(0).to(self.device, dtype=DTYPE)
        logits = self.model(frames)
        if isinstance(logits, tuple):
            logits = logits[0]
        if greedy or self.config.get("algo_type") not in (None, "POLICY"):
            return int(logits.argmax(-1).item())
        return int(Categorical(logits=logits).sample().item())


def run_episode(agent: Agent, seed: int, greedy: bool = False) -> float:
    env = make_env(seed, agent.config)
    obs = env.reset()
    score = 0.0
    done = False
    while not done:
        frames = obs[0]
        action = agent.select_action(frames, greedy=greedy)
        obs, reward, done, _ = env.step(action)
        score += float(reward[0])
    env.close()
    return score


def latest_model() -> Path | None:
    runs = sorted((ROOT / "runs").glob("run_*/model.onnx"), key=os.path.getmtime)
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DOOM model or checkpoint.")
    parser.add_argument(
        "--submission",
        default="",
        help="Path to a run directory, model.onnx, or .pth checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--greedy", action="store_true", help="Use argmax actions.")
    parser.add_argument("--device", default=str(DEVICE), help="Torch device, e.g. cuda or cpu.")
    args = parser.parse_args()
    device = torch.device(args.device)

    model_path = resolve_target(args.submission)
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            "No model found. Pass --submission path/to/run, model.onnx, or checkpoint.pth"
        )
    os.chdir(ROOT)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model, config = load_policy(model_path, device)
    agent = Agent(model, config, device)

    print(f"Model : {model_path}")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, sort_keys=True)}")

    scores = []
    for _ in range(args.episodes):
        ep_seed = int(np.random.randint(int(1e7)))
        scores.append(run_episode(agent, seed=ep_seed, greedy=args.greedy))

    print(f"Scores: {[round(s, 2) for s in scores]}")
    mean = float(np.mean(scores)) if scores else 0.0
    std = float(np.std(scores)) if scores else 0.0
    print(f"Mean  : {mean:.4f}")
    print(f"Std   : {std:.4f}")


if __name__ == "__main__":
    main()
