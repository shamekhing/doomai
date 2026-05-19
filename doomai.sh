#!/usr/bin/env bash
# One entrypoint for every workflow in this repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    VENV="$ROOT/.venv"
else
    VENV="$ROOT/../.venv"
fi
PY="$VENV/bin/python"

MODE="train"
TARGET=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
doomai.sh

Usage:
  ./doomai.sh MODE [TARGET] [OPTIONS]

Core RL workflows:
  train, ppo, --train, --ppo
      Train PPO with train.py defaults.

  dqn, --dqn
      Train a DQN/QVALUE agent.

  hybrid, --hybrid
      DQN warm-up, copy compatible weights into PPO, then PPO fine-tune.

  goal, --goal
      Long PPO run:
        --total-steps 6000000 --rollout-steps 2048 --eval-episodes 30

  quick, --quick
      Tiny PPO smoke run:
        --total-steps 4096 --rollout-steps 512 --eval-episodes 1

Imitation-learning workflows:
  collect-demos, demos, --collect-demos
      Human-play demo collection.

  bc-pretrain, il, bc, --bc-pretrain
      Behavior clone a DQN initializer from saved demos.

  hybrid-demos [BC_CKPT.pth], il-hybrid [BC_CKPT.pth], --hybrid-demos [BC_CKPT.pth]
      Human demos -> behavior-cloned DQN -> DQN warm-up -> PPO fine-tune.
      If BC_CKPT.pth is provided, it is forwarded as --bc-checkpoint.

  export-bc-policy BC_CKPT.pth, --export-bc-policy BC_CKPT.pth
      Convert a behavior-cloned DQN checkpoint into an actor ONNX.

Run utilities:
  resume RUN_OR_CKPT, --resume RUN_OR_CKPT
      Continue PPO, DQN, or hybrid checkpoint/run.

  eval [RUN_OR_MODEL_OR_CKPT], --eval [RUN_OR_MODEL_OR_CKPT]
      Evaluate a run directory, model.onnx, or .pth checkpoint.
      If target is omitted, eval.py uses the newest runs/*/model.onnx.

  view RUN_OR_MODEL_OR_CKPT, --view RUN_OR_MODEL_OR_CKPT
      Record GIF/MP4 replay from a run, ONNX, or checkpoint.

  export RUN_OR_CKPT, export-onnx RUN_OR_CKPT, pth-to-onnx CKPT.pth
  --export RUN_OR_CKPT, --export-onnx RUN_OR_CKPT, --pth-to-onnx CKPT.pth
      Export a run directory or .pth checkpoint to ONNX.
      Use --out PATH to choose the output file. Without --out, checkpoints export
      to their run directory's model.onnx, matching train.py.

  package RUN_DIR, --package RUN_DIR
      Build doom_submission.zip with code.py and model.onnx.

Help utilities:
  help, -h, --help
      Show this help.

  train-help
      Show train.py help.

  eval-help
      Show eval.py help.

  view-help
      Show view.py help.

  il-help, doomai-help
      Show the imitation-learning CLI help.

Training options forwarded where supported:
  --algo {ppo,dqn,hybrid}
  --run-dir DIR
  --resume RUN_OR_CKPT
  --export RUN_OR_CKPT
  --out PATH
  --total-steps N
  --rollout-steps N
  --eval-episodes N
  --no-eval
  --seed N
  --device DEVICE
  --learning-rate LR
  --entropy-coef X
  --entropy-coef-final X
  --clip-coef X
  --target-kl X
  --epsilon-start X
  --epsilon-end X
  --epsilon-decay-steps N
  --hybrid-dqn-steps N
  --dqn-learning-rate LR
  --policy-init-scale X
  --movement-reward-coef X
  --idle-penalty X
  --attack-no-hit-penalty X
  --aim-center-reward-coef X
  --aim-attack-reward-coef X
  --attack-not-centered-penalty X

IL/demo options forwarded where supported:
  --demo-dir DIR
  --bc-checkpoint PATH
  --bc-epochs N
  --bc-batch-size N
  --keep-noop
  --drop-noop
  --balance-actions
  --bc-class-weight-cap X
  --auto-eval-view
  --auto-bc-eval-view
  --bc-eval-episodes N
  --bc-view
  --bc-view-out PATH
  --post-eval-episodes N
  --post-view
  --post-view-out PATH
  --post-greedy
  --episodes N
  --fps N
  --window-size N

Eval/view options forwarded where supported:
  --submission RUN_OR_MODEL_OR_CKPT
  --episodes N
  --seed N
  --greedy
  --device DEVICE
  --fps N
  --out PATH
  --subsample N

Examples:
  ./doomai.sh quick --device cpu
  ./doomai.sh ppo --total-steps 1000000 --device cuda
  ./doomai.sh dqn --total-steps 1000000 --device cuda
  ./doomai.sh hybrid --total-steps 1000000 --hybrid-dqn-steps 200000 --device cuda
  ./doomai.sh collect-demos --demo-dir demos/human --episodes 5
  ./doomai.sh bc-pretrain --demo-dir demos/human --bc-epochs 8 --device cuda
  ./doomai.sh hybrid-demos /any/path/bc-dqn-init.pth --total-steps 1000000 --hybrid-dqn-steps 200000 --device cuda
  ./doomai.sh hybrid-demos --demo-dir demos/human --auto-bc-eval-view --bc-eval-episodes 10 --device cuda
  ./doomai.sh hybrid-demos --demo-dir demos/human --auto-eval-view --post-eval-episodes 10 --device cuda
  ./doomai.sh export-bc-policy /any/path/bc-dqn-init.pth --out bc-policy.onnx --device cuda
  ./doomai.sh eval runs/run_.../ckpts/best-checkpoint.pth --episodes 50 --device cuda --greedy
  ./doomai.sh view runs/run_... --out replays/doom.gif --episodes 1 --device cuda
  ./doomai.sh export runs/run_.../ckpts/best-checkpoint.pth
  ./doomai.sh pth-to-onnx runs/run_.../ckpts/best-checkpoint.pth --out model.onnx
  ./doomai.sh package runs/run_...

Notes:
  - VizDoom uses local sockets. Sandboxed terminals may need permission to bind a local port.
  - The challenge submission is still model.onnx; .pth eval is for local checkpoints.
EOF
}

require_venv() {
    if [[ ! -x "$PY" ]]; then
        echo "ERROR: venv not found. Run: bash scripts/setup.sh" >&2
        exit 1
    fi
}

latest_model() {
    find "$ROOT/runs" -path '*/model.onnx' -type f -printf '%T@ %p\n' 2>/dev/null \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
}

needs_value() {
    case "$1" in
        --algo|--run-dir|--resume|--export|--export-onnx|--pth-to-onnx|\
        --to-onnx|--out|--total-steps|--rollout-steps|\
        --eval-episodes|--best-metric|--seed|--device|--learning-rate|--entropy-coef|\
        --entropy-coef-final|--clip-coef|--target-kl|--epsilon-start|\
        --epsilon-end|--epsilon-decay-steps|--hybrid-dqn-steps|\
        --dqn-learning-rate|--policy-init-scale|--movement-reward-coef|\
        --idle-penalty|--attack-no-hit-penalty|--aim-center-reward-coef|\
        --aim-attack-reward-coef|--attack-not-centered-penalty|\
        --demo-dir|--bc-checkpoint|\
        --bc-epochs|--bc-batch-size|--bc-class-weight-cap|\
        --episodes|--fps|--window-size|\
        --submission|--subsample|--bc-eval-episodes|--bc-view-out|\
        --post-eval-episodes|--post-view-out)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

set_mode() {
    case "$1" in
        train|ppo|--train|--ppo) MODE="train" ;;
        dqn|--dqn) MODE="dqn" ;;
        hybrid|--hybrid) MODE="hybrid" ;;
        goal|--goal) MODE="goal" ;;
        quick|--quick) MODE="quick" ;;
        collect-demos|demos|--collect-demos) MODE="collect-demos" ;;
        bc-pretrain|bc|il|--bc-pretrain) MODE="bc-pretrain" ;;
        hybrid-demos|il-hybrid|--hybrid-demos) MODE="hybrid-demos" ;;
        export-bc-policy|--export-bc-policy) MODE="export-bc-policy" ;;
        resume|--resume) MODE="resume" ;;
        eval|--eval) MODE="eval" ;;
        view|--view) MODE="view" ;;
        export|export-onnx|pth-to-onnx|to-onnx|onnx|onyx|\
        --export|--export-onnx|--pth-to-onnx|--to-onnx|--onnx|--onyx)
            MODE="export"
            ;;
        package|--package) MODE="package" ;;
        train-help) MODE="train-help" ;;
        eval-help) MODE="eval-help" ;;
        view-help) MODE="view-help" ;;
        il-help|doomai-help) MODE="il-help" ;;
        help|-h|--help) MODE="help" ;;
        *)
            return 1
            ;;
    esac
    return 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        if [[ "$1" =~ ^[[:space:]]+$ ]]; then
            echo "ERROR: got a whitespace-only argument." >&2
            echo "This usually means a line-continuation backslash has trailing spaces." >&2
            echo "Put the backslash as the final character on the line, or run the command on one line." >&2
            exit 1
        fi
        if set_mode "$1"; then
            case "$MODE" in
                resume|view|export|package|hybrid-demos|export-bc-policy)
                    if [[ $# -ge 2 && "${2:-}" != --* ]]; then
                        TARGET="$2"
                        shift 2
                    else
                        shift
                    fi
                    ;;
                eval)
                    if [[ $# -ge 2 && "${2:-}" != --* ]]; then
                        TARGET="$2"
                        shift 2
                    else
                        shift
                    fi
                    ;;
                *)
                    shift
                    ;;
            esac
        elif needs_value "$1"; then
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 needs a value." >&2
                exit 1
            fi
            case "$1" in
                --submission)
                    TARGET="$2"
                    ;;
                --resume)
                    MODE="resume"
                    TARGET="$2"
                    ;;
                --export|--export-onnx|--pth-to-onnx|--to-onnx|--onnx|--onyx)
                    MODE="export"
                    TARGET="$2"
                    ;;
                *)
                    EXTRA_ARGS+=("$1" "$2")
                    ;;
            esac
            shift 2
        else
            case "$1" in
                --greedy|--no-eval|--keep-noop|--drop-noop|--balance-actions|\
                --auto-eval-view|\
                --auto-bc-eval-view|--bc-view|--post-view|--post-greedy)
                    EXTRA_ARGS+=("$1")
                    shift
                    ;;
                --*)
                    EXTRA_ARGS+=("$1")
                    shift
                    ;;
                *)
                    if [[ -z "$TARGET" ]]; then
                        TARGET="$1"
                    else
                        EXTRA_ARGS+=("$1")
                    fi
                    shift
                    ;;
            esac
        fi
    done
}

has_extra_arg() {
    local needle="$1"
    local arg
    for arg in "${EXTRA_ARGS[@]}"; do
        [[ "$arg" == "$needle" ]] && return 0
    done
    return 1
}

add_bc_checkpoint_target() {
    if [[ -z "$TARGET" ]]; then
        return
    fi
    [[ -e "$TARGET" ]] || { echo "ERROR: BC checkpoint not found: $TARGET" >&2; exit 1; }
    if ! has_extra_arg "--bc-checkpoint"; then
        EXTRA_ARGS+=("--bc-checkpoint" "$TARGET")
    fi
}

run_package() {
    local run_dir="$1"
    local model="$run_dir/model.onnx"
    local out="$run_dir/doom_submission.zip"
    local tmp_dir

    [[ -n "$run_dir" ]] || { echo "ERROR: package needs RUN_DIR." >&2; exit 1; }
    [[ -d "$run_dir" ]] || { echo "ERROR: run dir not found: $run_dir" >&2; exit 1; }
    [[ -f "$model" ]] || { echo "ERROR: missing $model; run export first." >&2; exit 1; }

    tmp_dir="$(mktemp -d /tmp/doomai_package.XXXXXX)"
    cat > "$tmp_dir/code.py" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_original_path = list(sys.path)
sys.path = [
    p for p in sys.path
    if Path(p or os.getcwd()).resolve() != ROOT
]
import code as _stdlib_code  # noqa: F401

sys.path = _original_path
sys.path.insert(0, str(ROOT))

from core.rl import main  # noqa: E402


if __name__ == "__main__":
    main()
PY
    (
        cd "$ROOT"
        zip -j "$out" "$tmp_dir/code.py" "$model" >/dev/null
        zip -r "$out" __main__.py arena common core dqn hybrid il ppo utility \
          -i '*.py' -i 'arena/bots/*.cfg' -i 'arena/scenarios/*' >/dev/null
    )
    rm -rf "$tmp_dir"
    echo "Package written: $out"
}

main() {
    parse_args "$@"

    if [[ "$MODE" == "help" ]]; then
        usage
        exit 0
    fi

    require_venv
    cd "$ROOT"

    case "$MODE" in
        train)
            PYTHONPATH="$ROOT" "$PY" -m core.rl "${EXTRA_ARGS[@]}"
            ;;
        dqn)
            PYTHONPATH="$ROOT" "$PY" -m core.rl --algo dqn "${EXTRA_ARGS[@]}"
            ;;
        hybrid)
            PYTHONPATH="$ROOT" "$PY" -m core.rl --algo hybrid "${EXTRA_ARGS[@]}"
            ;;
        goal)
            PYTHONPATH="$ROOT" "$PY" -m core.rl --total-steps 6000000 --rollout-steps 2048 --eval-episodes 30 "${EXTRA_ARGS[@]}"
            ;;
        quick)
            PYTHONPATH="$ROOT" "$PY" -m core.rl --total-steps 4096 --rollout-steps 512 --eval-episodes 1 "${EXTRA_ARGS[@]}"
            ;;
        collect-demos)
            PYTHONPATH="$ROOT" "$PY" -m utility.cli --collect-demos "${EXTRA_ARGS[@]}"
            ;;
        bc-pretrain)
            PYTHONPATH="$ROOT" "$PY" -m utility.cli --bc-pretrain "${EXTRA_ARGS[@]}"
            ;;
        hybrid-demos)
            add_bc_checkpoint_target
            PYTHONPATH="$ROOT" "$PY" -m utility.cli --hybrid-from-demos "${EXTRA_ARGS[@]}"
            ;;
        export-bc-policy)
            add_bc_checkpoint_target
            PYTHONPATH="$ROOT" "$PY" -m utility.cli --export-bc-policy "${EXTRA_ARGS[@]}"
            ;;
        resume)
            [[ -n "$TARGET" ]] || { echo "ERROR: resume needs RUN_OR_CKPT." >&2; exit 1; }
            PYTHONPATH="$ROOT" "$PY" -m core.rl --resume "$TARGET" "${EXTRA_ARGS[@]}"
            ;;
        eval)
            if [[ -n "$TARGET" ]]; then
                [[ -e "$TARGET" ]] || { echo "ERROR: eval target not found: $TARGET" >&2; exit 1; }
                PYTHONPATH="$ROOT" "$PY" -m utility.eval --submission "$TARGET" "${EXTRA_ARGS[@]}"
            else
                PYTHONPATH="$ROOT" "$PY" -m utility.eval "${EXTRA_ARGS[@]}"
            fi
            ;;
        view)
            [[ -n "$TARGET" ]] || { echo "ERROR: view needs RUN_OR_MODEL_OR_CKPT." >&2; exit 1; }
            [[ -e "$TARGET" ]] || { echo "ERROR: view target not found: $TARGET" >&2; exit 1; }
            PYTHONPATH="$ROOT" "$PY" -m utility.view "$TARGET" "${EXTRA_ARGS[@]}"
            ;;
        export)
            [[ -n "$TARGET" ]] || { echo "ERROR: export needs RUN_OR_CKPT." >&2; exit 1; }
            [[ -e "$TARGET" ]] || { echo "ERROR: export target not found: $TARGET" >&2; exit 1; }
            PYTHONPATH="$ROOT" "$PY" -m core.rl --export "$TARGET" "${EXTRA_ARGS[@]}"
            ;;
        package)
            run_package "$TARGET"
            ;;
        train-help)
            PYTHONPATH="$ROOT" "$PY" -m core.rl --help
            ;;
        eval-help)
            PYTHONPATH="$ROOT" "$PY" -m utility.eval --help
            ;;
        view-help)
            PYTHONPATH="$ROOT" "$PY" -m utility.view --help
            ;;
        il-help)
            PYTHONPATH="$ROOT" "$PY" -m utility.cli --help
            ;;
        *)
            echo "ERROR: unknown mode: $MODE" >&2
            usage >&2
            exit 1
            ;;
    esac
}

main "$@"
