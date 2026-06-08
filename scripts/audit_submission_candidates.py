#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_values(path: Path) -> list[float]:
    if not path.exists():
        return []
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                values.append(float(row[2]))
            except ValueError:
                continue
    return values


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


@dataclass
class Candidate:
    model: Path
    run: Path
    best_mean: float | None
    last_mean: float | None
    best_max: float | None
    best_robust: float | None
    max_episode_reward: float | None

    def sort_value(self, key: str) -> float:
        value = {
            "mean": self.best_mean,
            "last": self.last_mean,
            "max": self.best_max,
            "robust": self.best_robust,
            "episode": self.max_episode_reward,
        }[key]
        return float("-inf") if value is None else value


def summarize_model(model: Path) -> Candidate:
    run = model.parent
    logs = run / "logs"
    eval_score = read_values(logs / "eval_score.log")
    eval_max = read_values(logs / "eval_max.log")
    eval_robust = read_values(logs / "eval_robust.log")
    episode_reward = read_values(logs / "episode_reward.log")
    return Candidate(
        model=model,
        run=run,
        best_mean=max(eval_score) if eval_score else None,
        last_mean=eval_score[-1] if eval_score else None,
        best_max=max(eval_max) if eval_max else None,
        best_robust=max(eval_robust) if eval_robust else None,
        max_episode_reward=max(episode_reward) if episode_reward else None,
    )


def find_candidates() -> list[Candidate]:
    models = sorted(ROOT.glob("runs/**/model.onnx")) + sorted(ROOT.glob("results/*.onnx"))
    return [summarize_model(model) for model in models]


def print_table(candidates: list[Candidate], limit: int) -> None:
    header = (
        "rank  best_mean  last_mean  best_max  best_robust  max_ep_reward  model"
    )
    print(header)
    print("-" * len(header))
    for rank, candidate in enumerate(candidates[:limit], start=1):
        rel = candidate.model.relative_to(ROOT)
        print(
            f"{rank:>4}  {fmt(candidate.best_mean):>9}  {fmt(candidate.last_mean):>9}  "
            f"{fmt(candidate.best_max):>8}  {fmt(candidate.best_robust):>11}  "
            f"{fmt(candidate.max_episode_reward):>13}  {rel}"
        )


def run_eval(model: Path, episodes: int, device: str, greedy: bool) -> int:
    cmd = [
        sys.executable,
        "-m",
        "utility.eval",
        "--submission",
        str(model),
        "--episodes",
        str(episodes),
        "--device",
        device,
    ]
    if greedy:
        cmd.append("--greedy")
    print("Eval command:", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank submission model candidates from logs and optionally evaluate one."
    )
    parser.add_argument("--sort", choices=["mean", "last", "max", "robust", "episode"], default="mean")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--evaluate", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = sorted(
        find_candidates(),
        key=lambda candidate: candidate.sort_value(args.sort),
        reverse=True,
    )
    print_table(candidates, args.limit)
    if args.evaluate is None:
        return 0
    model = args.evaluate
    if not model.is_absolute():
        model = ROOT / model
    return run_eval(model, args.episodes, args.device, args.greedy)


if __name__ == "__main__":
    raise SystemExit(main())
