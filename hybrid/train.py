from __future__ import annotations

import copy
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from core.rl import (
    ROOT,
    DoomQNetwork,
    export_onnx,
    init_actor_critic_from_q,
    latest_checkpoint,
    load_checkpoint,
    model_input_channels,
    save_checkpoint,
    set_seed,
    train,
    train_dqn,
)
from il.demos import pretrain_dqn_from_demos


def run_post_training_tools(args, run_dir: Path) -> None:
    model_path = run_dir / "model.onnx"
    if not model_path.exists():
        print(f"Skipping post-training eval/view; missing {model_path}.")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    eval_episodes = int(args.post_eval_episodes)
    post_view = bool(args.post_view)
    if args.auto_eval_view:
        eval_episodes = eval_episodes or 10
        post_view = True

    if eval_episodes > 0:
        cmd = [
            sys.executable,
            "-m",
            "utility.eval",
            "--submission",
            str(model_path),
            "--episodes",
            str(eval_episodes),
            "--device",
            str(args.device_obj),
        ]
        if args.post_greedy:
            cmd.append("--greedy")
        print("Post-training eval:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    if post_view:
        out = Path(args.post_view_out).resolve() if args.post_view_out else run_dir / "replay.gif"
        cmd = [
            sys.executable,
            "-m",
            "utility.view",
            str(model_path),
            "--out",
            str(out),
            "--episodes",
            "1",
            "--device",
            str(args.device_obj),
        ]
        if args.post_greedy:
            cmd.append("--greedy")
        print("Post-training view:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_bc_tools(args, bc_ckpt: Path, run_dir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    eval_episodes = int(args.bc_eval_episodes)
    bc_view = bool(args.bc_view)
    if args.auto_bc_eval_view:
        eval_episodes = eval_episodes or 10
        bc_view = True

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


def hybrid_from_demos(args, config: dict) -> None:
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else ROOT / "runs" / f"demo_hybrid_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    set_seed(config["seed"])
    bc_ckpt = Path(args.bc_checkpoint).resolve() if args.bc_checkpoint else None
    if bc_ckpt is None:
        print("Phase 1/3: behavior cloning from human demonstrations.")
        bc_ckpt = pretrain_dqn_from_demos(args, config, run_dir)
    if not bc_ckpt.exists():
        raise FileNotFoundError(bc_ckpt)
    run_bc_tools(args, bc_ckpt, run_dir)

    dqn_steps = min(int(config["hybrid_dqn_steps"]), int(config["total_steps"]))
    dqn_ckpt = bc_ckpt
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
        print(f"Phase 2/3: DQN warm-up from demos for {dqn_steps} step(s).")
        train_dqn(
            dqn_config,
            dqn_run_dir,
            resume=bc_ckpt,
            device=args.device_obj,
            overrides=dqn_config,
        )
        dqn_ckpt = latest_checkpoint(dqn_run_dir) or bc_ckpt

    q_model, _, dqn_ckpt_config, dqn_global_step, _ = load_checkpoint(dqn_ckpt, args.device_obj)
    if not isinstance(q_model, DoomQNetwork):
        raise TypeError(f"Expected DQN checkpoint, got {type(q_model).__name__}")

    ppo_config = copy.deepcopy(config)
    ppo_config["algo"] = "hybrid"
    direct_bc_to_ppo = dqn_steps <= 0 and (
        dqn_ckpt_config.get("bc_policy") or dqn_ckpt.name == "bc-dqn-init.pth"
    )
    if direct_bc_to_ppo:
        ppo_config["algo_type"] = "POLICY"
        if args.policy_init_scale is None:
            ppo_config["policy_init_scale"] = 1.0
        if args.learning_rate is None:
            ppo_config["learning_rate"] = 2e-5
        if args.entropy_coef is None:
            ppo_config["entropy_coef"] = 0.001
        if args.entropy_coef_final is None:
            ppo_config["entropy_coef_final"] = 0.0001
        print(
            "Direct BC -> PPO mode: preserving BC policy scale and using "
            "conservative PPO defaults. Override with --policy-init-scale, "
            "--learning-rate, --entropy-coef, or --entropy-coef-final."
        )
    in_ch = model_input_channels(ppo_config)
    model = init_actor_critic_from_q(
        q_model,
        in_ch,
        args.device_obj,
        policy_scale=float(ppo_config["policy_init_scale"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config["learning_rate"], eps=1e-5)
    init_ckpt = run_dir / "ckpts" / "demo-hybrid-ppo-init.pth"
    save_checkpoint(
        init_ckpt,
        model,
        optimizer,
        ppo_config,
        global_step=dqn_global_step,
        best_eval=-float("inf"),
    )
    print(f"Phase 3/3: PPO fine-tune initialized from {dqn_ckpt}.")
    train(ppo_config, run_dir, init_ckpt, args.device_obj, overrides=None)
    run_post_training_tools(args, run_dir)


def export_bc_policy(args, config: dict) -> None:
    ckpt = Path(args.bc_checkpoint).resolve()
    q_model, _, ckpt_config, _, _ = load_checkpoint(ckpt, args.device_obj)
    ckpt_config["algo_type"] = "POLICY"
    out = Path(args.out).resolve() if args.out else ckpt.with_suffix(".onnx")
    export_onnx(q_model, ckpt_config, out, args.device_obj)
