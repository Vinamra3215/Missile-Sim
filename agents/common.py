"""Shared training utilities: episode statistics, curriculum and CSV logging."""
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name="cpu"):
    return name


class Curriculum:
    """
    Raises the map difficulty level when the agent is good at the current one.
    Only episodes played near the top level count towards promotion.
    """

    def __init__(self, env, start=0.0, step=0.1, threshold=0.75, window=2000, enabled=True):
        self.env, self.step, self.threshold = env, step, threshold
        self.window = window
        self.enabled = enabled
        self.recent = deque(maxlen=window)
        env.set_level(start if enabled else env.cfg.max_level)
        if not enabled:
            env.level_spread = 0.0

    def update(self, ep):
        if not self.enabled or ep is None:
            return False
        top = ep["level"] >= self.env.level - self.step - 1e-6
        self.recent.extend(ep["success"][top].tolist())
        if (len(self.recent) == self.window and self.env.level < self.env.cfg.max_level - 1e-6
                and np.mean(self.recent) >= self.threshold):
            self.env.set_level(round(self.env.level + self.step, 3))
            self.recent.clear()
            return True
        return False


class Tracker:
    """Collects finished-episode stats between log points and writes a CSV."""

    FIELDS = ["global_step", "time_s", "episodes", "level", "return", "success",
              "crash", "timeout", "length", "success_top", "loss", "extra"]

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "progress.csv")
        self.f = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        self.w.writeheader()
        self.t0 = time.time()
        self.total_eps = 0
        self._reset()

    def _reset(self):
        self.buf = {k: [] for k in ["return", "success", "crash", "timeout", "length", "level"]}

    def add(self, ep):
        if ep is None:
            return
        for k in self.buf:
            self.buf[k].append(ep[k].float().cpu())
        self.total_eps += len(ep["return"])

    def log(self, global_step, level, loss=float("nan"), extra=""):
        if not self.buf["return"]:
            return None
        cat = {k: torch.cat(v) for k, v in self.buf.items()}
        top = cat["level"] >= level - 0.1 - 1e-6
        row = {
            "global_step": global_step,
            "time_s": round(time.time() - self.t0, 1),
            "episodes": self.total_eps,
            "level": level,
            "return": cat["return"].mean().item(),
            "success": cat["success"].mean().item(),
            "crash": cat["crash"].mean().item(),
            "timeout": cat["timeout"].mean().item(),
            "length": cat["length"].mean().item(),
            "success_top": cat["success"][top].mean().item() if top.any() else float("nan"),
            "loss": loss,
            "extra": extra,
        }
        self.w.writerow(row)
        self.f.flush()
        self._reset()
        return row

    def close(self):
        self.f.close()
