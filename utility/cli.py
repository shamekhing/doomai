from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from core.rl import DEVICE, HYBRID_CONFIG, ROOT, apply_overrides
from hybrid.train import export_bc_policy, hybrid_from_demos
from il.demos import collect_demos, pretrain_dqn_from_demos


def run_bc_tools(args, bc_ckpt: Path, run_dir: Path) -> None:
    eval_episodes = int(args.bc_eval_episodes)
    bc_view = bool(args.bc_view)
    if args.auto_bc_eval_view:
        eval_episodes = eval_episodes or 10
        bc_view = True

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    if eval_episodes > 0:
        cmd = [
            sys.executable,
            "-m",
            "utility.eval",
            "--submission",
            str(bc_ckpt),
            "--episodes",
            str(eval_episodes),
            "--device",
            str(args.device_obj),
        ]
        print("Post-BC eval:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    if bc_view:
        out = Path(args.bc_view_out).resolve() if args.bc_view_out else run_dir / "bc-replay.gif"
        cmd = [
            sys.executable,
            "-m",
            "utility.view",
            str(bc_ckpt),
            "--out",
            str(out),
            "--episodes",
            "1",
            "--device",
            str(args.device_obj),
        ]
        print("Post-BC view:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def make_hybrid_config(args) -> dict:
    config = HYBRID_CONFIG.copy()
    overrides = {
        "total_steps": args.total_steps,
        "rollout_steps": args.rollout_steps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "entropy_coef_final": args.entropy_coef_final,
        "hybrid_dqn_steps": args.hybrid_dqn_steps,
        "dqn_learning_rate": args.dqn_learning_rate,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "policy_init_scale": args.policy_init_scale,
        "movement_reward_coef": args.movement_reward_coef,
        "idle_penalty": args.idle_penalty,
        "attack_no_hit_penalty": args.attack_no_hit_penalty,
        "aim_center_reward_coef": args.aim_center_reward_coef,
        "aim_attack_reward_coef": args.aim_attack_reward_coef,
        "attack_not_centered_penalty": args.attack_not_centered_penalty,
        "no_eval": args.no_eval,
    }
    return apply_overrides(config, overrides)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect human demos and use them before DQN+PPO hybrid training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect-demos", action="store_true")
    mode.add_argument("--bc-pretrain", action="store_true")
    mode.add_argument("--hybrid-from-demos", action="store_true")
    mode.add_argument("--export-bc-policy", action="store_true")

    parser.add_argument("--demo-dir", default="demos/human")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--bc-checkpoint", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=35)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument(
        "--keep-noop",
        action="store_true",
        help="Deprecated compatibility flag; no-op actions are kept by default.",
    )
    parser.add_argument(
        "--drop-noop",
        action="store_true",
        help="Drop no-op actions from behavior cloning. Usually hurts timing-heavy demos.",
    )
    parser.add_argument(
        "--balance-actions",
        action="store_true",
        help="Oversample each present action equally during BC. Off by default to preserve demo behavior.",
    )
    parser.add_argument(
        "--bc-class-weight-cap",
        type=float,
        default=0.0,
        help="Enable inverse-frequency BC class weights capped at this value. 0 disables weights.",
    )
    parser.add_argument(
        "--auto-eval-view",
        action="store_true",
        help="After demo-hybrid training, run eval and record a replay from model.onnx.",
    )
    parser.add_argument(
        "--auto-bc-eval-view",
        action="store_true",
        help="After behavior cloning, run eval and record a replay before DQN/PPO.",
    )
    parser.add_argument(
        "--bc-eval-episodes",
        type=int,
        default=0,
        help="Evaluate the BC checkpoint after behavior cloning. 0 disables BC eval.",
    )
    parser.add_argument(
        "--bc-view",
        action="store_true",
        help="Record a replay from the BC checkpoint after behavior cloning.",
    )
    parser.add_argument(
        "--bc-view-out",
        default="",
        help="Replay output path for --bc-view. Defaults to RUN_DIR/bc-replay.gif.",
    )
    parser.add_argument(
        "--post-eval-episodes",
        type=int,
        default=0,
        help="Evaluate model.onnx after demo-hybrid training. 0 disables post eval.",
    )
    parser.add_argument(
        "--post-view",
        action="store_true",
        help="Record a replay from model.onnx after demo-hybrid training.",
    )
    parser.add_argument(
        "--post-view-out",
        default="",
        help="Replay output path for --post-view. Defaults to RUN_DIR/replay.gif.",
    )
    parser.add_argument(
        "--post-greedy",
        action="store_true",
        help="Use argmax actions for post-training eval/view.",
    )

    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--entropy-coef-final", type=float, default=None)
    parser.add_argument("--hybrid-dqn-steps", type=int, default=None)
    parser.add_argument("--dqn-learning-rate", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--policy-init-scale", type=float, default=None)
    parser.add_argument("--movement-reward-coef", type=float, default=None)
    parser.add_argument("--idle-penalty", type=float, default=None)
    parser.add_argument("--attack-no-hit-penalty", type=float, default=None)
    parser.add_argument("--aim-center-reward-coef", type=float, default=None)
    parser.add_argument("--aim-attack-reward-coef", type=float, default=None)
    parser.add_argument("--attack-not-centered-penalty", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device_obj = torch.device(args.device)
    config = make_hybrid_config(args)

    if args.collect_demos:
        collect_demos(args, config)
        return

    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else ROOT / "runs" / f"bc_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.bc_pretrain:
        bc_ckpt = pretrain_dqn_from_demos(args, config, run_dir)
        run_bc_tools(args, bc_ckpt, run_dir)
    elif args.hybrid_from_demos:
        hybrid_from_demos(args, config)
    elif args.export_bc_policy:
        export_bc_policy(args, config)


if __name__ == "__main__":
    main()
