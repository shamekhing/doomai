# doomai

`doomai` is a standalone deep reinforcement learning project for the JKU Doom challenge. It trains agents in a ViZDoom deathmatch arena using PPO, DQN, a DQN-to-PPO hybrid pipeline, and an optional human-demonstration pipeline that starts with behavior cloning.

The repository includes the Doom arena configuration, bot settings, training loops, demo collection tools, local evaluation, replay recording, and ONNX export for challenge-style submissions.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [If Human Demo Pretraining Is Already Done](#if-human-demo-pretraining-is-already-done)
- [Action Space](#action-space)
- [Human Demonstrations](#human-demonstrations)
- [Training Workflows](#training-workflows)
- [Evaluation](#evaluation)
- [Replay Recording](#replay-recording)
- [Outputs and Checkpoints](#outputs-and-checkpoints)
- [Configuration](#configuration)
- [Code Architecture](#code-architecture)
- [Direct Python Entry Points](#direct-python-entry-points)
- [Troubleshooting](#troubleshooting)

## Project Overview

The project trains an agent in `arena/scenarios/jku.wad` with the scenario configuration in `arena/scenarios/jku.cfg`. Training uses a single controlled player against four easy bots on the `ROOM` map by default.

Supported workflows:

- **PPO**: trains an actor-critic policy directly from environment interaction.
- **DQN**: trains a Q-network with replay buffer and epsilon-greedy exploration.
- **Hybrid DQN -> PPO**: trains DQN first, copies compatible weights into a PPO actor-critic, then fine-tunes with PPO.
- **Demo Hybrid**: behavior-clones a DQN initializer from human demos, performs DQN warm-up, then fine-tunes with PPO.
- **Evaluation and viewing**: evaluates exported ONNX models locally and records GIF/MP4 replays.

The training code exports `model.onnx` with metadata describing the observation format and policy type expected by the local evaluator.

## Repository Structure

```text
doomai/
  arena/                 ViZDoom environment wrapper, player setup, bots, scenario files
    bots/                Easy and hard bot configuration files
    scenarios/           jku.cfg and jku.wad
  common/                Shared action names and display helpers
  core/                  PPO, DQN, hybrid training, models, checkpointing, ONNX export
  demos/human/           Saved human demonstration .pt files
  dqn/                   DQN-facing exports around core.rl
  hybrid/                Demo + DQN + PPO pipeline helpers
  il/                    Human demo collection and behavior cloning
  ppo/                   PPO-facing exports around core.rl
  scripts/               Setup and runner shell scripts
  utility/               CLI, ONNX evaluation, replay/video recording
  requirements.txt       Python dependencies
  README.md              This documentation
```

Important files:

- `core/rl.py`: main models, PPO loop, DQN loop, hybrid loop, checkpointing, evaluation during training, and ONNX export.
- `utility/cli.py`: CLI for demo collection, behavior cloning, and demo-based hybrid training.
- `il/demos.py`: human demo recording, demo dataset loading, and behavior-cloning pretraining.
- `hybrid/train.py`: behavior-cloning -> DQN warm-up -> PPO fine-tuning pipeline.
- `utility/eval.py`: local ONNX evaluator.
- `utility/view.py`: records trained agents to GIF or MP4.
- `arena/doom_env.py`: Gymnasium-style ViZDoom multiplayer environment wrapper.
- `arena/player.py`: player/bot/game setup and observation-buffer options.
- `common/actions.py`: discrete action IDs and keyboard/mouse mapping for demo collection.

## Requirements

The project is Python-based and expects ViZDoom and PyTorch.

Main dependencies:

- `python3`
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `tqdm`
- `vizdoom`
- `gymnasium`
- `pygame`
- `onnx`
- `onnxscript`
- `onnx2pytorch`
- `imageio`
- `imageio-ffmpeg`
- `portpicker`

The setup script creates a virtual environment with `--system-site-packages`. This is intentional: it allows the environment to use system-installed CUDA/PyTorch packages when available.

## Setup

From the `doomai/` project root:

```bash
bash scripts/setup.sh
```

The script performs three steps:

1. Creates `.venv` with system site packages.
2. Installs the Python dependencies.
3. Registers a Jupyter kernel named `Python (DRL-Doom)`.

Activate the environment manually when needed:

```bash
source .venv/bin/activate
```

Check that CUDA is available:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Training defaults to `--device cuda`. If CUDA is unavailable, pass `--device cpu`, but expect training to be much slower.

## Quick Start

Collect five human demonstration episodes:

```bash
./doomai.sh collect-demos --episodes 5 --demo-dir demos/human --device cuda
```

Train the recommended demo-based hybrid pipeline from demos:

```bash
./doomai.sh hybrid-demos \
  --demo-dir demos/human \
  --bc-epochs 3 \
  --total-steps 1000000 \
  --hybrid-dqn-steps 200000 \
  --device cuda \
  --auto-bc-eval-view \
  --bc-eval-episodes 10
```

`--auto-bc-eval-view` evaluates and records a replay immediately after Phase 1 behavior cloning, before DQN warm-up and PPO fine-tuning. This is the fastest way to check whether the cloned policy is doing anything sensible before spending time on the full run.

If you already have a human-trained/behavior-cloned `.pth`, pass that checkpoint as the argument:

```bash
./doomai.sh hybrid-demos /any/path/to/bc-dqn-init.pth \
  --demo-dir demos/human \
  --total-steps 1000000 \
  --hybrid-dqn-steps 200000 \
  --device cuda
```

Evaluate the exported model:

```bash
./doomai.sh --eval runs/RUN_NAME/model.onnx --episodes 50 --device cuda
```

Record a replay:

```bash
./doomai.sh --view runs/RUN_NAME/model.onnx --out replay.gif --episodes 1 --device cuda
```

Replace `RUN_NAME` with the run directory created under `runs/`.

## If Human Demo Pretraining Is Already Done

Use this section when you already played/recorded human episodes or already trained the behavior-cloned DQN checkpoint.

If you only have human demo `.pt` files, first train the behavior-cloned DQN initializer:

```bash
./doomai.sh bc-pretrain \
  --demo-dir demos/human \
  --bc-epochs 8 \
  --bc-batch-size 64 \
  --device cuda
```

That produces:

```text
runs/bc_YYYY_MM_DD-HH_MM_SS/ckpts/bc-dqn-init.pth
```

If you already have the behavior-cloned DQN checkpoint, do not recollect demos and do not redo behavior cloning unless you want a new initializer. Continue from that `.pth` into the full demo hybrid pipeline. The checkpoint can be anywhere on disk:

```bash
./doomai.sh hybrid-demos /any/path/to/bc-dqn-init.pth \
  --demo-dir demos/human \
  --total-steps 1000000 \
  --hybrid-dqn-steps 200000 \
  --device cuda
```

This positional checkpoint argument is the same as `--bc-checkpoint /any/path/to/bc-dqn-init.pth`; the positional form is shorter and preferred.

That runs:

```text
existing human-demo BC DQN checkpoint
  -> DQN warm-up in the Doom arena
  -> PPO actor-critic initialized from DQN
  -> PPO fine-tuning
  -> final model.onnx
```

For a fast sanity check before a long run, use fewer steps and skip periodic eval:

```bash
./doomai.sh hybrid-demos /any/path/to/bc-dqn-init.pth \
  --demo-dir demos/human \
  --total-steps 50000 \
  --hybrid-dqn-steps 10000 \
  --no-eval \
  --device cuda
```

Important: when splitting commands across lines, the `\` must be the last character on the line. Do not write `\ ` with a trailing space. This is also valid as one line:

```bash
./doomai.sh hybrid-demos /any/path/to/bc-dqn-init.pth --demo-dir demos/human --total-steps 50000 --hybrid-dqn-steps 10000 --no-eval --device cuda
```

The hybrid demo run writes to:

```text
runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/
```

After the hybrid run starts producing checkpoints, you can evaluate any `.pth` immediately without waiting for ONNX:

```bash
./doomai.sh eval runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/ckpts/best-checkpoint.pth \
  --episodes 50 \
  --device cuda \
  --greedy
```

Export a checkpoint to ONNX when you are ready to submit or compare through ONNX:

```bash
./doomai.sh pth-to-onnx \
  runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/ckpts/best-checkpoint.pth \
  --out runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/model.onnx \
  --device cuda
```

Then run the final ONNX eval and record a replay:

```bash
./doomai.sh eval runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/model.onnx \
  --episodes 50 \
  --device cuda \
  --greedy

./doomai.sh view runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/model.onnx \
  --out replay.gif \
  --episodes 1 \
  --device cuda
```

Use the generated `runs/demo_hybrid_.../model.onnx` as the challenge model. The `.pth` files are for local resume/eval/export.

To automate both inspection points in one command, add:

```bash
--auto-bc-eval-view --bc-eval-episodes 10 \
--auto-eval-view --post-eval-episodes 10
```

That records:

```text
runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/bc-replay.gif
runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/replay.gif
```

## Action Space

The environment uses a single discrete action at each step. The configured buttons are defined by `arena/scenarios/jku.cfg` and mapped in `common/actions.py`.

| ID | Action |
| --- | --- |
| `0` | `noop` |
| `1` | `move_forward` |
| `2` | `attack` |
| `3` | `move_left` |
| `4` | `move_right` |
| `5` | `turn_left` |
| `6` | `turn_right` |
| `7` | `jump` |

Because the action space is single-action discrete, demos and policies cannot currently express simultaneous combinations such as moving forward while attacking.

## Human Demonstrations

### Controls

Used by `--collect-demos`:

```text
SPACE / mouse attack
LEFT          turn left
RIGHT         turn right
A             strafe left
D             strafe right
W / UP         move forward
J             jump
ESC           quit
```

When multiple keys are held, collection records one discrete action using this priority: attack, turn, strafe, forward, jump, no-op. This matters because the model can only imitate one action per frame.

### Collect Demos

```bash
./doomai.sh --collect-demos \
  --episodes 5 \
  --demo-dir demos/human \
  --fps 35 \
  --window-size 512 \
  --device cuda
```

Each episode is saved as:

```text
demos/human/demo_YYYY_MM_DD-HH_MM_SS_epNNN.pt
```

Each `.pt` demo stores:

- `obs`: observation tensors saved as `float16`.
- `actions`: integer action IDs.
- `rewards`: reward values.
- `dones`: episode done flags.
- `config`: observation/export settings used during collection.
- `action_names`: action ID to name mapping.

### Behavior Clone From Demos

```bash
./doomai.sh --bc-pretrain \
  --demo-dir demos/human \
  --bc-epochs 3 \
  --bc-batch-size 64 \
  --device cuda
```

Output:

```text
runs/bc_YYYY_MM_DD-HH_MM_SS/ckpts/bc-dqn-init.pth
```

By default, behavior cloning filters out no-op actions. To keep no-op samples:

```bash
./doomai.sh --bc-pretrain --demo-dir demos/human --keep-noop --device cuda
```

Behavior cloning uses class-weighted cross entropy so rare actions such as attack/turn/strafe are not drowned out by frequent forward movement. The loader streams one demo file at a time, so shuffled BC training does not repeatedly reload large `.pt` files.

To inspect a behavior-cloned policy immediately inside the demo-hybrid workflow:

```bash
./doomai.sh hybrid-demos \
  --demo-dir demos/human \
  --bc-epochs 3 \
  --total-steps 1000000 \
  --hybrid-dqn-steps 200000 \
  --device cuda \
  --auto-bc-eval-view \
  --bc-eval-episodes 10
```

After Phase 1, this evaluates `bc-dqn-init.pth` and records:

```text
runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/bc-replay.gif
```

### Export A Behavior-Cloned Policy

If you have a behavior-cloned DQN checkpoint and want to export it as an ONNX policy:

```bash
./doomai.sh export-bc-policy /any/path/to/bc-dqn-init.pth \
  --out runs/bc_YYYY_MM_DD-HH_MM_SS/bc-policy.onnx \
  --device cuda
```

## Training Workflows

The recommended interface is `./doomai.sh`, which selects the correct root script or `doomai` package workflow.

### PPO From Scratch

```bash
./doomai.sh --train \
  --total-steps 1000000 \
  --device cuda
```

This runs `core.rl` with the default PPO configuration:

- actor-critic model
- stacked grayscale frames with labels/depth extra state
- PPO clipped objective
- GAE advantage estimation
- periodic checkpointing
- periodic local evaluation
- ONNX export

### DQN From Scratch

```bash
./doomai.sh --dqn \
  --total-steps 1000000 \
  --device cuda
```

This runs a Q-network with:

- replay buffer
- epsilon-greedy exploration
- target network updates
- Huber loss
- greedy evaluation
- ONNX export with `QVALUE` metadata

### DQN -> PPO Hybrid

```bash
./doomai.sh --hybrid \
  --total-steps 1000000 \
  --hybrid-dqn-steps 200000 \
  --device cuda
```

Pipeline:

```text
DQN warm-up
  -> copy compatible encoder/trunk/action weights into PPO actor-critic
  -> PPO fine-tuning
  -> model.onnx export
```

The DQN warm-up output is saved inside:

```text
runs/run_YYYY_MM_DD-HH_MM_SS/dqn_warmup/
```

### Human Demo Hybrid

```bash
./doomai.sh --hybrid-demos \
  --demo-dir demos/human \
  --bc-epochs 3 \
  --bc-batch-size 64 \
  --total-steps 1000000 \
  --hybrid-dqn-steps 200000 \
  --device cuda \
  --auto-bc-eval-view \
  --bc-eval-episodes 10
```

Pipeline:

```text
human demos
  -> behavior-cloned DQN initializer
  -> optional BC eval and bc-replay.gif
  -> DQN warm-up
  -> copy DQN weights into PPO actor-critic
  -> PPO fine-tuning
  -> model.onnx export
  -> optional final eval and replay.gif
```

To reuse an existing behavior-cloned checkpoint:

```bash
./doomai.sh --hybrid-demos /any/path/to/bc-dqn-init.pth \
  --demo-dir demos/human \
  --total-steps 1000000 \
  --hybrid-dqn-steps 200000 \
  --device cuda
```

Post-Phase-1 inspection flags:

```bash
--auto-bc-eval-view      # run BC eval and BC replay after behavior cloning
--bc-eval-episodes 10   # number of BC eval episodes
--bc-view               # record only the BC replay
--bc-view-out PATH      # choose BC replay output path
```

Post-training inspection flags:

```bash
--auto-eval-view        # run final eval and final replay after model.onnx export
--post-eval-episodes 10 # number of final eval episodes
--post-view             # record only the final replay
--post-view-out PATH    # choose final replay output path
--post-greedy           # use argmax actions for final eval/view
```

### Resume Training

Resume from a run directory:

```bash
./doomai.sh --train --resume runs/run_YYYY_MM_DD-HH_MM_SS --device cuda
```

Resume from a specific checkpoint:

```bash
./doomai.sh --train \
  --resume runs/run_YYYY_MM_DD-HH_MM_SS/ckpts/checkpoint-step25000.pth \
  --device cuda
```

When a directory is passed to `--resume`, the code tries `ckpts/best-checkpoint.pth` first, then the latest `checkpoint-step*.pth`.

### Export A Checkpoint To ONNX

```bash
./doomai.sh pth-to-onnx \
  runs/run_YYYY_MM_DD-HH_MM_SS/ckpts/best-checkpoint.pth \
  --out runs/run_YYYY_MM_DD-HH_MM_SS/model.onnx \
  --device cuda
```

This accepts a direct `.pth` checkpoint or a run directory. No training occurs during export.

## Evaluation

Evaluate an ONNX model:

```bash
./doomai.sh --eval runs/run_YYYY_MM_DD-HH_MM_SS/model.onnx \
  --episodes 50 \
  --seed 1337 \
  --device cuda
```

Use greedy action selection:

```bash
./doomai.sh --eval runs/run_YYYY_MM_DD-HH_MM_SS/model.onnx \
  --episodes 50 \
  --greedy \
  --device cuda
```

The evaluator:

- loads ONNX metadata key `config`
- builds the same `ROOM` arena with four easy bots
- uses categorical sampling for `POLICY` models
- uses argmax for `QVALUE` models or when `--greedy` is passed
- prints per-episode scores, mean, and standard deviation

## Replay Recording

Record a run directory, ONNX model, or checkpoint:

```bash
./doomai.sh --view runs/run_YYYY_MM_DD-HH_MM_SS \
  --out replay.gif \
  --episodes 1 \
  --fps 20 \
  --device cuda
```

Supported targets:

- a run directory containing `model.onnx`
- a direct `.onnx` file
- a direct `.pth` checkpoint

Examples:

```bash
./doomai.sh --view runs/run_YYYY_MM_DD-HH_MM_SS/model.onnx --out replay.gif
./doomai.sh --view runs/run_YYYY_MM_DD-HH_MM_SS/ckpts/best-checkpoint.pth --out replay.mp4
```

Useful options:

- `--episodes`: number of episodes to record.
- `--fps`: output GIF/MP4 frames per second.
- `--subsample`: keep every Nth frame to reduce file size.
- `--greedy`: use argmax action selection.

## Outputs and Checkpoints

Training creates a run directory under `runs/` unless `--run-dir` is provided.

Typical PPO/DQN run:

```text
runs/run_YYYY_MM_DD-HH_MM_SS/
  ckpts/
    checkpoint-step25000.pth
    best-checkpoint.pth
    final-checkpoint.pth
  logs/
    episode_reward.log
    episode_length.log
    loss.log
    eval_score.log
    eval_robust.log
  training_curves.png
  model.onnx
```

Typical demo hybrid run:

```text
runs/demo_hybrid_YYYY_MM_DD-HH_MM_SS/
  ckpts/
    bc-dqn-init.pth
    demo-hybrid-ppo-init.pth
    checkpoint-step*.pth
    best-checkpoint.pth
    final-checkpoint.pth
  dqn_warmup/
    ckpts/
    logs/
    training_curves.png
    model.onnx
  logs/
  training_curves.png
  model.onnx
  bc-replay.gif       # if --auto-bc-eval-view or --bc-view was used
  replay.gif          # if --auto-eval-view or --post-view was used
```

Checkpoint files store:

- model state dict
- optimizer state dict
- training config
- global step
- best evaluation metric
- checkpoint kind such as `ppo`, `dqn`, or `hybrid`

`model.onnx` includes a metadata key named `config` containing:

- `screen_format`
- `n_stack_frames`
- `extra_state`
- `hud`
- `crosshair`
- `algo_type`

## Configuration

Default configs are defined in `core/rl.py`:

- `CONFIG`: PPO defaults.
- `DQN_CONFIG`: DQN defaults.
- `HYBRID_CONFIG`: hybrid defaults.

Important defaults:

| Setting | PPO Default | DQN Default | Hybrid Default |
| --- | ---: | ---: | ---: |
| `total_steps` | `1000000` | `1000000` | `1000000` |
| `learning_rate` | `2e-4` | `1e-4` | `2e-4` |
| `gamma` | `0.99` | `0.95` | `0.99` |
| `eval_episodes` | `30` | `30` | `30` |
| `checkpoint_interval` | `25000` | `25000` | `25000` |
| `num_bots` | `4` | `4` | `4` |
| `bot_skill` | `0` | `0` | `0` |
| `doom_map` | `ROOM` | `ROOM` | `ROOM` |
| `n_stack_frames` | `4` | `1` | `4` |
| `algo_type` | `ARGMAX` | `QVALUE` | `ARGMAX` |
| `hybrid_dqn_steps` | n/a | n/a | `200000` |

Common CLI overrides:

```bash
--total-steps N
--rollout-steps N
--eval-episodes N
--no-eval
--seed N
--device cuda
--learning-rate LR
--entropy-coef VALUE
--entropy-coef-final VALUE
--clip-coef VALUE
--target-kl VALUE
--epsilon-start VALUE
--epsilon-end VALUE
--epsilon-decay-steps N
--hybrid-dqn-steps N
--dqn-learning-rate LR
--policy-init-scale VALUE
```

Demo-hybrid inspection options:

```bash
--auto-bc-eval-view
--bc-eval-episodes N
--bc-view
--bc-view-out PATH
--auto-eval-view
--post-eval-episodes N
--post-view
--post-view-out PATH
--post-greedy
```

Training-only reward shaping options:

```bash
--movement-reward-coef VALUE
--idle-penalty VALUE
--attack-no-hit-penalty VALUE
```

The code uses challenge-aligned evaluation rewards for scoring, while training can add small shaping signals to reduce behavior such as standing still or repeatedly attacking without hits.

## Code Architecture

### Environment Layer

`arena/doom_env.py` wraps ViZDoom as Gymnasium-style environments:

- `PlayerEnv`: one player instance, action processing, observation collection, frame stacking, and game variable tracking.
- `VizdoomMPEnv`: multiplayer wrapper that hosts the player, adds bots, exposes reset/step, and supports replay capture.

`arena/player.py` configures:

- ViZDoom game mode and map
- available buttons
- available game variables
- HUD/crosshair rendering
- depth/labels/automap buffers
- multiplayer host/join settings
- bot config paths

### Observation Format

The model input is built from:

- the screen buffer
- optional `labels`
- optional `depth`
- optional `automap`
- optional frame stacking

PPO/hybrid default:

```text
screen_format = GRAY8
n_stack_frames = 4
extra_state = ["labels", "depth"]
```

DQN default:

```text
screen_format = GRAY8
n_stack_frames = 1
extra_state = ["depth"]
```

### Models

`DoomActorCritic`:

- convolutional encoder
- 512-unit trunk
- policy logits head
- value head
- used for PPO and final hybrid fine-tuning

`DoomQNetwork`:

- convolutional encoder
- 512-unit head
- action-value output
- used for DQN and behavior cloning

`ActorExport` wraps either model type for ONNX export.

### PPO

Implemented in `core/rl.py`:

- collects rollouts with `rollout_steps`
- computes GAE advantages
- applies clipped PPO policy loss
- applies value loss and entropy regularization
- linearly interpolates entropy coefficient from `entropy_coef` to `entropy_coef_final`
- checkpoints and evaluates periodically

### DQN

Implemented in `core/rl.py`:

- stores transitions in `ReplayBuffer`
- starts training after `train_after`
- trains every `train_every` environment steps
- uses Double-DQN-style next action selection
- updates target network every `target_update_interval`
- decays epsilon from `epsilon_start` to `epsilon_end`

### Hybrid

The hybrid path:

1. Trains or loads a DQN model.
2. Copies compatible DQN encoder and trunk weights into the actor-critic model.
3. Copies the DQN action head into the PPO policy head with `policy_init_scale`.
4. Fine-tunes the actor-critic with PPO.

### Demonstration Learning

`il/demos.py` provides:

- `collect_demos`: pygame-based human control loop.
- `save_demo_episode`: writes `.pt` demos.
- `DemoDataset`: indexes all `.pt` demos in a directory.
- `pretrain_dqn_from_demos`: supervised cross-entropy training from observations to actions.

## Direct Python Entry Points

The `./doomai.sh` wrapper is preferred. Direct module calls also work from the `doomai/` project root:

Collect demos:

```bash
PYTHONPATH=. .venv/bin/python -m utility.cli --collect-demos --episodes 5 --demo-dir demos/human --device cuda
```

Behavior clone:

```bash
PYTHONPATH=. .venv/bin/python -m utility.cli --bc-pretrain --demo-dir demos/human --device cuda
```

Demo hybrid:

```bash
PYTHONPATH=. .venv/bin/python -m utility.cli --hybrid-from-demos --demo-dir demos/human --device cuda
```

PPO:

```bash
PYTHONPATH=. .venv/bin/python -m core.rl --algo ppo --total-steps 1000000 --device cuda
```

DQN:

```bash
PYTHONPATH=. .venv/bin/python -m core.rl --algo dqn --total-steps 1000000 --device cuda
```

Hybrid:

```bash
PYTHONPATH=. .venv/bin/python -m core.rl --algo hybrid --total-steps 1000000 --hybrid-dqn-steps 200000 --device cuda
```

Evaluate:

```bash
PYTHONPATH=. .venv/bin/python -m utility.eval --submission runs/RUN_NAME/model.onnx --episodes 50 --device cuda
```

Record replay:

```bash
PYTHONPATH=. .venv/bin/python -m utility.view runs/RUN_NAME/model.onnx --out replay.gif --device cuda
```

## Troubleshooting

### CUDA is not available

Check PyTorch:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, install a CUDA-enabled PyTorch build or use `--device cpu` for small tests.

### ViZDoom cannot open sockets or ports

The environment uses local multiplayer hosting and `portpicker`. Sandboxed terminals or restricted environments may block local socket binding. Run from a normal terminal if ViZDoom fails during game initialization.

### `--no-eval: command not found`

This is a shell line-continuation problem, not a Python option problem. It happens when a line ends with backslash plus a trailing space:

```bash
--hybrid-dqn-steps 10000 \ 
```

Use either one line, or make the backslash the final character:

```bash
--hybrid-dqn-steps 10000 \
--no-eval \
--device cuda
```

### Demo collection window does not appear

`pygame` needs access to a display server. Run demo collection in a desktop session rather than a headless terminal.

### No demo files found

Behavior cloning expects `.pt` files in `--demo-dir`:

```bash
ls demos/human/*.pt
```

Collect demos first or pass the correct demo directory.

### Training seems slow

Use CUDA and reduce expensive periodic evaluation while iterating:

```bash
./doomai.sh --hybrid-demos /any/path/to/bc-dqn-init.pth --demo-dir demos/human --total-steps 100000 --no-eval --device cuda
```

For final runs, re-enable evaluation and use more episodes for a more stable estimate.

### ONNX export fails

Ensure the ONNX dependencies are installed:

```bash
.venv/bin/pip install onnx onnxscript onnx2pytorch
```

Then export from a checkpoint:

```bash
./doomai.sh pth-to-onnx runs/RUN_NAME/ckpts/best-checkpoint.pth --device cuda
```

### Replay output is too large

Increase subsampling or reduce episodes:

```bash
./doomai.sh --view runs/RUN_NAME/model.onnx --out replay.gif --episodes 1 --subsample 4
```

## Notes

- `arena/` is the local ViZDoom/Arena adaptation used by this project.
- `utility/` contains the user-facing evaluation and replay tools.
- `core/rl.py` is intentionally central: it keeps model definitions, training loops, checkpoint format, and ONNX export in one place.
- Human demonstrations are useful for bootstrapping behavior, but the current action space only records one action per frame.
- Long evaluations are more reliable than short ones because Doom episodes have high score variance.
