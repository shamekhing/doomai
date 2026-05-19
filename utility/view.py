"""
Record a trained DOOM agent.

Usage:
  python -m utility.view runs/run_YYYY_MM_DD-HH_MM_SS
  python -m utility.view runs/run_.../model.onnx --episodes 3
  python -m utility.view runs/run_.../ckpts/best-checkpoint.pth --out replay.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-step(\d+)\.pth", path.name)
    return int(match.group(1)) if match else -1


def resolve_target(target: str) -> Path:
    path = Path(target).resolve()
    if path.is_dir():
        onnx_path = path / "model.onnx"
        if onnx_path.exists():
            return onnx_path
        best = path / "ckpts" / "best-checkpoint.pth"
        if best.exists():
            return best
        ckpts = list((path / "ckpts").glob("checkpoint-step*.pth"))
        if ckpts:
            return max(ckpts, key=checkpoint_step)
    return path


def read_onnx_config(model_proto) -> dict:
    for prop in model_proto.metadata_props:
        if prop.key == "config":
            return json.loads(prop.value)
    return {
        "screen_format": 8,
        "n_stack_frames": 4,
        "extra_state": None,
        "hud": "minimal",
        "crosshair": True,
        "algo_type": "POLICY",
    }


def load_policy(path: Path):
    if path.suffix == ".onnx":
        proto = onnx.load(path)
        config = read_onnx_config(proto)
        model = ConvertModel(proto).to(DEVICE).eval()
        return model, config

    model, _, config, _, _ = load_checkpoint(path, DEVICE)
    model.eval()
    export_config = onnx_config(config)
    if config.get("bc_policy") or path.name == "bc-dqn-init.pth":
        export_config["algo_type"] = "POLICY"
    return model, export_config


def make_env(config: dict, seed: int) -> VizdoomMPEnv:
    extra_state = None
    if config["extra_state"]:
        extra_state = [ObsBuffer(x) for x in config["extra_state"]]
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


@torch.no_grad()
def select_action(model: torch.nn.Module, frames: torch.Tensor, config: dict, greedy: bool):
    logits = model(frames.unsqueeze(0).to(DEVICE, dtype=torch.float32))
    if isinstance(logits, tuple):
        logits = logits[0]
    if greedy or config.get("algo_type") not in (None, "POLICY"):
        return int(logits.argmax(-1).item())
    return int(Categorical(logits=logits).sample().item())


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Install imageio to save videos: pip install imageio") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved replay: {out_path}")


def run_episode(model, config: dict, seed: int, greedy: bool):
    env = make_env(config, seed)
    env.enable_replay()
    obs = env.reset()
    score = 0.0
    done = False
    while not done:
        action = select_action(model, obs[0], config, greedy)
        obs, reward, done, _ = env.step(action)
        score += float(reward[0])
    replay = env.get_player_replays()["WhoAmI"]
    env.close()
    frames = [frame_to_rgb(f) for f in replay["frames"]]
    return score, frames


def main() -> None:
    global DEVICE
    parser = argparse.ArgumentParser(description="Record a DOOM agent replay.")
    parser.add_argument("target", help="Run directory, .onnx model, or .pth checkpoint.")
    parser.add_argument("--out", default="", help="Output .gif/.mp4 path.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--device", default=str(DEVICE), help="Torch device, e.g. cuda or cpu.")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)

    target = resolve_target(args.target)
    if not target.exists():
        raise FileNotFoundError(target)
    out = Path(args.out).resolve() if args.out else target.with_suffix(".gif")
    os.chdir(ROOT)

    model, config = load_policy(target)
    print(f"Model : {target}")
    print(f"Output: {out}")

    all_frames = []
    scores = []
    for ep in range(args.episodes):
        score, frames = run_episode(model, config, seed=args.seed + ep, greedy=args.greedy)
        scores.append(score)
        all_frames.extend(frames[:: max(args.subsample, 1)])
        print(f"Episode {ep + 1}: score={score:.2f}, frames={len(frames)}")

    save_video(all_frames, out, fps=args.fps)
    print(f"Mean score: {float(np.mean(scores)):.2f}")


if __name__ == "__main__":
    main()
