from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


# This file is named code.py, which can shadow Python's stdlib code module while
# PyTorch imports pdb. Load the stdlib module first, then restore the path.
SUBMISSION_DIR = Path(__file__).resolve().parent
_original_sys_path = list(sys.path)
sys.path = [
    item
    for item in sys.path
    if Path(item or os.getcwd()).resolve() != SUBMISSION_DIR
]
import code as _stdlib_code  # noqa: E402,F401

sys.modules["code"] = _stdlib_code
sys.path = _original_sys_path

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from onnx2pytorch import ConvertModel  # noqa: E402
from torch.distributions import Categorical  # noqa: E402


ACTION_NAMES = {
    0: "noop",
    1: "move_forward",
    2: "attack",
    3: "move_left",
    4: "move_right",
    5: "turn_left",
    6: "turn_right",
    7: "jump",
}


@dataclass
class BCConfig:
    screen_format: int = 8
    n_stack_frames: int = 4
    extra_state: tuple[str, ...] = ("labels", "depth")
    hud: str = "full"
    crosshair: bool = True
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5


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
            batch, channels, frames, height, width = x.shape
            x = x.reshape(batch, channels * frames, height, width)
        return x.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flatten_stack(x)
        return self.head(self.encoder(x))


class ActorExport(nn.Module):
    def __init__(self, model: DoomQNetwork):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.model(frames)


def input_channels(config: BCConfig) -> int:
    screen_channels = 1 if config.screen_format == 0 else 3
    buffer_channels = len(config.extra_state)
    return (screen_channels + buffer_channels) * config.n_stack_frames


def load_demo(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def demo_paths(demo_dir: Path) -> list[Path]:
    paths = sorted(demo_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt demos found in {demo_dir}")
    return paths


def action_counts(paths: list[Path], keep_noop: bool = True) -> torch.Tensor:
    counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
    for path in paths:
        actions = load_demo(path)["actions"].long()
        if not keep_noop:
            actions = actions[actions != 0]
        counts += torch.bincount(actions, minlength=len(ACTION_NAMES))[: len(ACTION_NAMES)]
    return counts


def summarize_demos(demo_dir: Path, keep_noop: bool = True) -> None:
    paths = demo_paths(demo_dir)
    counts = action_counts(paths, keep_noop=keep_noop)
    total_steps = int(counts.sum().item())
    reward_sum = 0.0
    for path in paths:
        data = load_demo(path)
        reward_sum += float(data.get("rewards", torch.tensor([])).float().sum().item())
    print(f"Demo files: {len(paths)}")
    print(f"Training steps: {total_steps}")
    print(f"Recorded reward: {reward_sum:.2f}")
    for action_id, count in enumerate(counts.tolist()):
        pct = 100.0 * count / max(total_steps, 1)
        print(f"{ACTION_NAMES[action_id]}: {count} ({pct:.2f}%)")


def iter_demo_batches(
    paths: list[Path],
    batch_size: int,
    keep_noop: bool = True,
    shuffle_files: bool = True,
):
    order = torch.randperm(len(paths)).tolist() if shuffle_files else list(range(len(paths)))
    for path_index in order:
        data = load_demo(paths[path_index])
        obs = data["obs"]
        actions = data["actions"].long()
        indices = torch.arange(actions.numel())
        if not keep_noop:
            indices = torch.nonzero(actions != 0).flatten()
        if indices.numel() == 0:
            continue
        indices = indices[torch.randperm(indices.numel())]
        for start in range(0, indices.numel(), batch_size):
            batch = indices[start : start + batch_size]
            yield obs[batch].float(), actions[batch]


def train_behavior_clone(
    demo_dir: Path,
    out_ckpt: Path,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    keep_noop: bool = True,
) -> Path:
    config = BCConfig()
    paths = demo_paths(demo_dir)
    model = DoomQNetwork(input_channels(config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        correct = 0
        total = 0
        model.train()
        for obs, actions in iter_demo_batches(paths, batch_size, keep_noop=keep_noop):
            obs = obs.to(device)
            actions = actions.to(device)
            logits = model(obs)
            loss = F.cross_entropy(logits, actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            losses.append(float(loss.item()))
            pred = logits.argmax(1)
            correct += int((pred == actions).sum().item())
            total += int(actions.numel())
        mean_loss = float(np.mean(losses)) if losses else 0.0
        accuracy = correct / max(total, 1)
        print(f"epoch={epoch} loss={mean_loss:.4f} acc={accuracy:.3f}")

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "dqn",
            "model": model.state_dict(),
            "config": {
                "algo": "dqn",
                "algo_type": "POLICY",
                "bc_policy": True,
                "screen_format": config.screen_format,
                "n_stack_frames": config.n_stack_frames,
                "extra_state": list(config.extra_state),
                "hud": config.hud,
                "crosshair": config.crosshair,
                "learning_rate": config.learning_rate,
            },
        },
        out_ckpt,
    )
    return out_ckpt


def export_onnx(ckpt_path: Path, out_path: Path, device: torch.device) -> Path:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    bc_config = BCConfig(
        screen_format=int(cfg["screen_format"]),
        n_stack_frames=int(cfg["n_stack_frames"]),
        extra_state=tuple(cfg["extra_state"]),
        hud=str(cfg["hud"]),
        crosshair=bool(cfg["crosshair"]),
    )
    model = DoomQNetwork(input_channels(bc_config)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros((1, 3, bc_config.n_stack_frames, 128, 128), device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        ActorExport(model),
        dummy,
        out_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["frames"],
        output_names=["logits"],
        external_data=False,
    )
    proto = onnx.load(out_path)
    del proto.metadata_props[:]
    meta = proto.metadata_props.add()
    meta.key = "config"
    meta.value = json.dumps(
        {
            "screen_format": bc_config.screen_format,
            "n_stack_frames": bc_config.n_stack_frames,
            "extra_state": list(bc_config.extra_state),
            "hud": bc_config.hud,
            "crosshair": bc_config.crosshair,
            "algo_type": "POLICY",
        }
    )
    onnx.save(proto, out_path)
    return out_path


def load_policy(model_path: Path, device: torch.device):
    proto = onnx.load(model_path)
    onnx.checker.check_model(proto)
    config = {}
    for item in proto.metadata_props:
        if item.key == "config":
            config = json.loads(item.value)
            break
    model = ConvertModel(proto).to(device).eval()
    return model, config


@torch.no_grad()
def select_action(model: nn.Module, frames: torch.Tensor, greedy: bool = False) -> int:
    if frames.ndim == 4:
        frames = frames.unsqueeze(0)
    logits = model(frames.float())
    if isinstance(logits, tuple):
        logits = logits[0]
    if greedy:
        return int(logits.argmax(-1).item())
    return int(Categorical(logits=logits).sample().item())


def inspect_model(model_path: Path, device: torch.device, greedy: bool) -> None:
    model, config = load_policy(model_path, device)
    print(f"Loaded model: {model_path}")
    print(f"Config: {json.dumps(config, sort_keys=True)}")
    dummy = torch.zeros((1, 3, int(config.get("n_stack_frames", 4)), 128, 128), device=device)
    action = select_action(model, dummy, greedy=greedy)
    print(f"Dummy action: {action} ({ACTION_NAMES[action]})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained Doom IL pipeline.")
    sub = parser.add_subparsers(dest="cmd")

    summary = sub.add_parser("summary")
    summary.add_argument("--demo-dir", required=True)
    summary.add_argument("--drop-noop", action="store_true")

    train = sub.add_parser("train-bc")
    train.add_argument("--demo-dir", required=True)
    train.add_argument("--out", default="bc-dqn-init.pth")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--drop-noop", action="store_true")

    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", default="model.onnx")
    export.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--model", default="model.onnx")
    inspect.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    inspect.add_argument("--greedy", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "summary":
        summarize_demos(Path(args.demo_dir), keep_noop=not args.drop_noop)
    elif args.cmd == "train-bc":
        train_behavior_clone(
            Path(args.demo_dir),
            Path(args.out),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=torch.device(args.device),
            keep_noop=not args.drop_noop,
        )
    elif args.cmd == "export":
        out = export_onnx(Path(args.checkpoint), Path(args.out), torch.device(args.device))
        print(f"Exported: {out}")
    else:
        model_path = Path(getattr(args, "model", "model.onnx"))
        if not model_path.is_absolute():
            model_path = SUBMISSION_DIR / model_path
        inspect_model(model_path, torch.device(getattr(args, "device", "cpu")), getattr(args, "greedy", False))


if __name__ == "__main__":
    main()
