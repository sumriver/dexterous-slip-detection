"""MuJoCo offscreen rendering and MP4 export."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np


def make_scene_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """Side view framing desk, bottle, and Shadow Hand."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.55, 0.0, 0.88])
    cam.distance = 1.35
    cam.azimuth = 132.0
    cam.elevation = -12.0
    return cam


class VideoRecorder:
    """Capture MuJoCo frames and write an MP4 file."""

    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: Path,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        timestep: float | None = None,
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.timestep = timestep if timestep is not None else model.opt.timestep
        self.frame_interval = max(1, int(round(1.0 / (fps * self.timestep))))
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = make_scene_camera(model)
        self._frames: list[np.ndarray] = []

    def maybe_capture(self, data: mujoco.MjData, step: int) -> None:
        if step % self.frame_interval != 0:
            return
        self.renderer.update_scene(data, camera=self.camera)
        self._frames.append(self.renderer.render().copy())

    def save(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._frames:
            raise RuntimeError("No frames captured; simulation may be too short.")
        iio.imwrite(
            self.output_path,
            np.stack(self._frames, axis=0),
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
        return self.output_path

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def close(self) -> None:
        self.renderer.close()
