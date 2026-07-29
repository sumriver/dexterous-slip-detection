"""NN-1 online detector: ring buffer + z-score + SlipTCN/GRU."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sim.slip_nn_features import FEATURE_DIM
from sim.slip_nn_model import DEFAULT_WINDOW, build_slip_model

# Teacher-leak / case-id channels zeroed when drop_leak_features=True.
IDX_SLIP_RULE_S2 = 17
IDX_GRIP_EXTRA = 23
IDX_PHASE_EXTEND = 24
IDX_FRICTION_SCALE = 25
LEAK_FEATURE_INDICES = (IDX_SLIP_RULE_S2, IDX_PHASE_EXTEND, IDX_FRICTION_SCALE)
# Multitask also drops proprio grip channel so the grip head cannot copy state.
LEAK_FEATURE_INDICES_MULTITASK = LEAK_FEATURE_INDICES + (IDX_GRIP_EXTRA,)


@dataclass
class SlipNnReading:
    p_slip: float
    slip_active: bool  # grip trigger (includes optional latch)
    n_valid_steps: int
    slip_now: bool = False  # raw p > τ this step (for false-trigger metrics)
    delta_grip: float | None = None  # optional NN-2 grip head output in [0, max]
    policy_grip: float | None = None  # optional Policy-1/2 target grip in [0, max]
    wrist_delta: tuple[float, float, float] | None = None  # Policy-2 wrist residual (rad)


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any] | Path) -> "NormStats":
        if isinstance(manifest, Path):
            manifest = json.loads(manifest.read_text())
        norm = manifest.get("norm") or {}
        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
            raise ValueError(f"norm dim mismatch: mean={mean.shape} std={std.shape}")
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) - self.mean) / self.std).astype(np.float32)


class SlipNeuralDetector:
    """Maintain T-step feature window; emit p_slip each update."""

    def __init__(
        self,
        model_path: str | Path,
        norm: NormStats | dict[str, Any] | Path,
        *,
        threshold: float = 0.5,
        device: str = "cpu",
        window_steps: int = DEFAULT_WINDOW,
        arch: str | None = None,
        latch: bool | None = None,
        drop_leak_features: bool | None = None,
        confirm_steps: int = 1,
        use_grip_head: bool | None = None,
        leak_indices: tuple[int, ...] | None = None,
        soft_threshold: float | None = None,
        soft_grip_scale: float | None = None,
        policy_mode: str | None = None,
        policy_width: int | None = None,
        max_grip: float | None = None,
        max_wrist: float | None = None,
        wrist_scale: float | None = None,
        residual: bool | None = None,
    ):
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.window_steps = int(window_steps)
        self.norm = norm if isinstance(norm, NormStats) else NormStats.from_manifest(norm)
        self.confirm_steps = max(1, int(confirm_steps))

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
            arch = arch or ckpt.get("arch", "tcn")
            feature_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
            if latch is None:
                latch = bool(ckpt.get("deploy_latch", False))
            if drop_leak_features is None:
                drop_leak_features = bool(ckpt.get("drop_leak_features", False))
            self.confirm_steps = max(1, int(ckpt.get("confirm_steps", self.confirm_steps)))
            if use_grip_head is None:
                use_grip_head = bool(
                    ckpt.get(
                        "use_grip_head",
                        arch
                        in (
                            "tcn_multi",
                            "multitask",
                            "detect_and_policy",
                            "detect_and_policy2",
                        ),
                    )
                )
            if soft_threshold is None:
                soft_threshold = float(ckpt.get("soft_threshold", 1.01))
            if soft_grip_scale is None:
                soft_grip_scale = float(ckpt.get("soft_grip_scale", 1.0))
            if policy_width is None and "policy_width" in ckpt:
                policy_width = int(ckpt["policy_width"])
            if max_grip is None and "max_grip" in ckpt:
                max_grip = float(ckpt["max_grip"])
            if max_wrist is None and "max_wrist" in ckpt:
                max_wrist = float(ckpt["max_wrist"])
            if wrist_scale is None and "wrist_scale" in ckpt:
                wrist_scale = float(ckpt["wrist_scale"])
            if residual is None and "residual" in ckpt:
                residual = bool(ckpt["residual"])
        else:
            state = ckpt
            arch = arch or "tcn"
            feature_dim = FEATURE_DIM
            if latch is None:
                latch = False
            if drop_leak_features is None:
                drop_leak_features = False
            if use_grip_head is None:
                use_grip_head = arch in (
                    "tcn_multi",
                    "multitask",
                    "detect_and_policy",
                    "detect_and_policy2",
                )
            if soft_threshold is None:
                soft_threshold = 1.01
            if soft_grip_scale is None:
                soft_grip_scale = 1.0

        self.arch = arch
        is_policy2 = arch == "detect_and_policy2"
        is_policy1 = arch == "detect_and_policy"
        is_policy = is_policy1 or is_policy2
        # Policy heads inherit NN-2 deploy defaults when ckpt/meta omit them.
        if is_policy:
            if latch is False and "deploy_latch" not in (ckpt if isinstance(ckpt, dict) else {}):
                latch = True
            if self.confirm_steps <= 1 and isinstance(ckpt, dict) and "confirm_steps" not in ckpt:
                self.confirm_steps = 30
            if soft_threshold is None or soft_threshold >= 1.0:
                soft_threshold = 0.7
            if self.threshold == 0.5:
                # Caller / meta may override; 0.5 is the ctor default before meta apply.
                pass

        if policy_mode is None:
            if is_policy2:
                policy_mode = "p2a"
            elif is_policy1:
                policy_mode = "replace"
            else:
                policy_mode = "off"
        policy_mode = str(policy_mode).lower()
        if policy_mode not in ("off", "replace", "residual", "p2a"):
            raise ValueError(
                f"policy_mode must be off|replace|residual|p2a, got {policy_mode!r}"
            )

        self.latch = bool(latch)
        self.drop_leak_features = bool(drop_leak_features)
        self.use_grip_head = bool(use_grip_head) or is_policy
        self.soft_threshold = float(soft_threshold)
        self.soft_grip_scale = float(soft_grip_scale)
        self.policy_mode = policy_mode
        self.use_policy = is_policy and policy_mode != "off"
        self.max_wrist = float(max_wrist if max_wrist is not None else 0.25)
        # Closed-loop often needs a softer wrist than open-loop teachers (BC shift).
        if wrist_scale is None:
            wrist_scale = 0.5 if is_policy2 else 1.0
        self.wrist_scale = float(wrist_scale)
        if leak_indices is not None:
            self.leak_indices = tuple(leak_indices)
        elif self.use_grip_head or is_policy:
            self.leak_indices = LEAK_FEATURE_INDICES_MULTITASK
        else:
            self.leak_indices = LEAK_FEATURE_INDICES

        if is_policy2:
            from sim.slip_nn_policy import DEFAULT_POLICY_WIDTH
            from sim.slip_nn_policy2 import SlipDetectAndPolicy2

            self.model = SlipDetectAndPolicy2(
                feature_dim=feature_dim,
                max_grip=float(max_grip if max_grip is not None else 0.25),
                max_wrist=self.max_wrist,
                policy_width=int(policy_width if policy_width is not None else DEFAULT_POLICY_WIDTH),
            )
        elif is_policy1:
            from sim.slip_nn_policy import DEFAULT_POLICY_WIDTH, SlipDetectAndPolicy

            self.model = SlipDetectAndPolicy(
                feature_dim=feature_dim,
                max_grip=float(max_grip if max_grip is not None else 0.25),
                residual=bool(residual) if residual is not None else False,
                policy_width=int(policy_width if policy_width is not None else DEFAULT_POLICY_WIDTH),
            )
        else:
            self.model = build_slip_model(arch, feature_dim=feature_dim)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self._buf: list[np.ndarray] = []
        self._latched = False
        self._high_run = 0

    def reset_extend(self) -> None:
        self._buf.clear()
        self._latched = False
        self._high_run = 0

    def reset(self) -> None:
        self.reset_extend()

    @property
    def n_valid_steps(self) -> int:
        return len(self._buf)

    def update(self, features: np.ndarray) -> SlipNnReading:
        """features: (D,) raw (unnormalized) current-step vector."""
        feat = np.asarray(features, dtype=np.float32).reshape(-1)
        if feat.shape[0] != FEATURE_DIM:
            raise ValueError(f"expected features ({FEATURE_DIM},), got {feat.shape}")

        # Capture proprio grip before leak-zero (policy head conditions on it).
        grip_raw = float(feat[IDX_GRIP_EXTRA])

        if self.drop_leak_features:
            feat = feat.copy()
            for idx in self.leak_indices:
                feat[idx] = 0.0

        self._buf.append(self.norm.transform(feat))
        if len(self._buf) > self.window_steps:
            self._buf.pop(0)

        n = len(self._buf)
        if n < self.window_steps:
            pad = [self._buf[0]] * (self.window_steps - n)
            window = np.stack(pad + self._buf, axis=0)
        else:
            window = np.stack(self._buf, axis=0)

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)
        delta_grip: float | None = None
        policy_grip: float | None = None
        wrist_delta: tuple[float, float, float] | None = None
        with torch.no_grad():
            if self.arch == "detect_and_policy2" and hasattr(self.model, "forward_policy"):
                grip_t = torch.tensor([grip_raw], dtype=torch.float32, device=self.device)
                p_t, g_ref, g_pol, w = self.model.forward_policy(x, grip_t)
                p = float(p_t.item())
                delta_grip = float(g_ref.item())
                policy_grip = float(g_pol.item())
                wrist_delta = tuple(float(v) for v in w[0].detach().cpu().tolist())
            elif self.arch == "detect_and_policy" and hasattr(self.model, "forward_policy"):
                grip_t = torch.tensor([grip_raw], dtype=torch.float32, device=self.device)
                p_t, g_ref, g_pol = self.model.forward_policy(x, grip_t)
                p = float(p_t.item())
                delta_grip = float(g_ref.item())
                policy_grip = float(g_pol.item())
            elif self.use_grip_head and hasattr(self.model, "forward_multi"):
                logit, grip = self.model.forward_multi(x)
                p = float(torch.sigmoid(logit).item())
                delta_grip = float(grip.item())
            else:
                p = float(self.model.predict_proba(x).item())
        raw_high = p > self.threshold
        if raw_high:
            self._high_run += 1
        else:
            self._high_run = 0
        fired = self._high_run >= self.confirm_steps
        if self.latch and fired:
            self._latched = True
        active = bool(self._latched if self.latch else fired)
        return SlipNnReading(
            p_slip=p,
            slip_active=active,
            n_valid_steps=n,
            slip_now=fired,
            delta_grip=delta_grip,
            policy_grip=policy_grip,
            wrist_delta=wrist_delta,
        )

    def resolve_grip(self, reading: SlipNnReading) -> float | None:
        """Pick grip command from reading given ``policy_mode``."""
        if self.policy_mode in ("replace", "p2a") and reading.policy_grip is not None:
            return float(reading.policy_grip)
        if self.policy_mode == "residual":
            if reading.delta_grip is None or reading.policy_grip is None:
                return reading.policy_grip
            return float(reading.delta_grip) + float(reading.policy_grip)
        if reading.delta_grip is not None:
            return float(reading.delta_grip)
        return None

    def resolve_wrist(self, reading: SlipNnReading) -> tuple[float, float, float]:
        """Pick wrist residual; zeros unless Policy-2 mode."""
        if self.policy_mode == "p2a" and reading.wrist_delta is not None:
            m = float(self.max_wrist)
            s = float(self.wrist_scale)
            wr, wp, wy = reading.wrist_delta
            return (
                float(np.clip(wr * s, -m, m)),
                float(np.clip(wp * s, -m, m)),
                float(np.clip(wy * s, -m, m)),
            )
        return (0.0, 0.0, 0.0)


def load_detector_from_dir(
    model_dir: Path,
    *,
    threshold: float | None = None,
    device: str = "cpu",
    policy_mode: str | None = None,
) -> SlipNeuralDetector:
    """Load ``slip_tcn_v1.pt`` / ``slip_policy_v1.pt`` / ``slip_policy2_v1.pt`` + sibling train_meta norm."""
    model_dir = Path(model_dir)
    pt = model_dir / "slip_tcn_v1.pt"
    if not pt.exists():
        for name in ("slip_policy2_v1.pt", "slip_policy_v1.pt"):
            candidate = model_dir / name
            if candidate.exists():
                pt = candidate
                break
        else:
            pts = sorted(model_dir.glob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"No checkpoint in {model_dir}")
            pt = pts[0]
    meta_path = model_dir / "train_meta.json"
    norm: NormStats | Path
    arch = None
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        arch = meta.get("arch")
        if "norm" in meta:
            norm = NormStats(
                mean=np.asarray(meta["norm"]["mean"], dtype=np.float32),
                std=np.asarray(meta["norm"]["std"], dtype=np.float32),
            )
        else:
            norm = Path(meta.get("manifest", "data/slip_nn/manifest.json"))
    else:
        norm = Path("data/slip_nn/manifest.json")

    is_policy = arch in ("detect_and_policy", "detect_and_policy2")
    if threshold is None:
        # Policy heads default to NN-2 hard τ=0.99 when meta omits it.
        default_thr = 0.99 if is_policy else 0.5
        threshold = float(meta.get("default_threshold", default_thr))
    use_grip = meta.get("use_grip_head")
    if use_grip is None and arch in (
        "tcn_multi",
        "multitask",
        "detect_and_policy",
        "detect_and_policy2",
    ):
        use_grip = True
    soft_thr = meta.get("soft_threshold")
    if soft_thr is None and is_policy:
        soft_thr = 0.7
    soft_scale = meta.get("soft_grip_scale")
    if soft_scale is None and is_policy:
        soft_scale = 1.0
    if policy_mode is None:
        policy_mode = meta.get("policy_mode")
    latch = meta.get("deploy_latch")
    if latch is None and is_policy:
        latch = True
    confirm = meta.get("confirm_steps")
    if confirm is None and is_policy:
        confirm = 30

    return SlipNeuralDetector(
        pt,
        norm,
        threshold=threshold,
        device=device,
        arch=arch,
        latch=bool(latch) if latch is not None else None,
        confirm_steps=int(confirm) if confirm is not None else 1,
        use_grip_head=bool(use_grip) if use_grip is not None else None,
        soft_threshold=float(soft_thr) if soft_thr is not None else None,
        soft_grip_scale=float(soft_scale) if soft_scale is not None else None,
        policy_mode=policy_mode,
        policy_width=int(meta["policy_width"]) if "policy_width" in meta else None,
        max_grip=float(meta["max_grip"]) if "max_grip" in meta else None,
        max_wrist=float(meta["max_wrist"]) if "max_wrist" in meta else None,
        wrist_scale=float(meta["wrist_scale"]) if "wrist_scale" in meta else None,
        residual=bool(meta["residual"]) if "residual" in meta else None,
    )
