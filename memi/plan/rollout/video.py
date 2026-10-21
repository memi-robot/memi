"""The episode video, the front and wrist cameras side by side, written as the env returns each frame so that a long
episode never sits in memory. It is for viewing only, so a failure to write it is noted in the log and the episode
goes on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: The simulator's control rate.
FPS = 20
KEYS = ("front_rgb_list", "wrist_rgb_list")


def _image(x) -> np.ndarray:
    """A frame as uint8 HxWx3 numpy, from a tensor or an array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.dtype != np.uint8:
        x = np.clip(x * 255 if x.max() <= 1.0 else x, 0, 255).astype(np.uint8)
    return x


class VideoWriter:
    """Appends the frames of each observation batch to one mp4."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.writer = None
        self.frames = 0
        self.error = ""

    def add(self, obs) -> None:
        if self.path is None or self.error or not isinstance(obs, dict):
            return
        try:
            front, wrist = (obs.get(k) for k in KEYS)
            front = [] if front is None else front
            wrist = [] if wrist is None else wrist
            for i, f in enumerate(front):
                img = _image(f)
                if i < len(wrist):
                    w = _image(wrist[i])
                    if w.shape[0] == img.shape[0]:
                        img = np.concatenate([img, w], axis=1)
                self._write(img)
        except Exception as exc:
            self.error = f"video: {type(exc).__name__}: {exc}"

    def _write(self, img: np.ndarray) -> None:
        import cv2
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          FPS, (w, h))
            self.size = (w, h)
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size)
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
