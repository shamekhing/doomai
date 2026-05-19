from __future__ import annotations

import numpy as np
import torch


def obs_to_display(obs: torch.Tensor) -> np.ndarray:
    frame = obs.detach().cpu()
    if frame.ndim == 4:
        frame = frame[0, -1]
    elif frame.ndim == 3:
        frame = frame[0]
    frame_np = frame.numpy()
    frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(frame_np[..., None], 3, axis=-1)

