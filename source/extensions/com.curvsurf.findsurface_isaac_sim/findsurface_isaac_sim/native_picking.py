"""Zero-copy native point picking with an explicit packaged-library boundary."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
import os
import platform
from pathlib import Path
from typing import Sequence

import numpy as np


class _NativePickResult(ctypes.Structure):
    _fields_ = (
        ("point_index", ctypes.c_int32),
        ("depth_m", ctypes.c_double),
        ("ray_distance_sq_m", ctypes.c_double),
        ("used_fallback", ctypes.c_uint8),
    )


@dataclass(frozen=True)
class NativePickResult:
    point_index: int
    depth_m: float
    ray_distance_m: float
    used_fallback: bool


class NativePointPicker:
    """Call the product-owned C ABI without copying immutable ScanFrame points."""

    def __init__(self, library_path: str | Path | None = None) -> None:
        self.library_path = Path(library_path) if library_path else default_library_path()
        if not self.library_path.is_file():
            raise FileNotFoundError(f"native picking library was not found: {self.library_path}")
        self._library = ctypes.CDLL(str(self.library_path))
        function = self._library.fsis_pick_perspective
        function.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.POINTER(_NativePickResult),
        )
        function.restype = ctypes.c_int
        self._pick_perspective = function
        self._pick_parallel = getattr(self._library, "fsis_pick_parallel", None)
        if self._pick_parallel is not None:
            self._pick_parallel.argtypes = function.argtypes
            self._pick_parallel.restype = ctypes.c_int

    @property
    def supports_parallel(self):
        return self._pick_parallel is not None

    def pick_perspective(
        self,
        points: np.ndarray,
        origin: Sequence[float],
        direction: Sequence[float],
        cone_half_angle_radians: float,
        *,
        fallback_to_nearest: bool,
        active_mask: Sequence[bool] | np.ndarray | None = None,
        parallel_radius_m: float | None = None,
    ) -> NativePickResult | None:
        if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3 or not points.flags.c_contiguous:
            raise ValueError("native picker requires C-contiguous float32[N,3] points")
        if points.shape[0] == 0:
            return None
        origin_values = (ctypes.c_double * 3)(*(float(value) for value in origin))
        direction_values = (ctypes.c_double * 3)(*(float(value) for value in direction))
        tangent = math.tan(float(cone_half_angle_radians)) if parallel_radius_m is None else float(parallel_radius_m)
        function = self._pick_perspective if parallel_radius_m is None else self._pick_parallel
        if function is None:
            raise RuntimeError("packaged picker does not support parallel rays")
        mask_array = None
        mask_pointer = None
        if active_mask is not None:
            mask_array = np.ascontiguousarray(active_mask, dtype=np.uint8)
            if mask_array.shape != (points.shape[0],):
                raise ValueError("active_mask must contain one boolean per point")
            mask_pointer = mask_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        result = _NativePickResult()
        found = function(
            ctypes.c_void_p(points.ctypes.data),
            mask_pointer,
            ctypes.c_int32(points.shape[0]),
            origin_values,
            direction_values,
            ctypes.c_double(tangent),
            ctypes.c_int32(1 if fallback_to_nearest else 0),
            ctypes.byref(result),
        )
        if not found:
            return None
        return NativePickResult(
            int(result.point_index),
            float(result.depth_m),
            math.sqrt(max(0.0, float(result.ray_distance_sq_m))),
            bool(result.used_fallback),
        )


def default_library_path() -> Path:
    override = os.environ.get("FSIS_PICKING_LIBRARY_PATH")
    if override:
        return Path(override).expanduser().resolve()
    data_root = Path(__file__).resolve().parents[1] / "data" / "native"
    system = platform.system().lower()
    if system == "windows":
        return data_root / "windows-x86_64" / "FindSurfaceIsaacSimPicking.dll"
    if system == "linux":
        return data_root / "linux-x86_64" / "libFindSurfaceIsaacSimPicking.so"
    raise RuntimeError(f"unsupported native picking platform: {platform.system()}")


_default_picker: NativePointPicker | None = None
_default_picker_attempted = False


def default_picker() -> NativePointPicker | None:
    global _default_picker, _default_picker_attempted
    if not _default_picker_attempted:
        _default_picker_attempted = True
        try:
            _default_picker = NativePointPicker()
        except (FileNotFoundError, OSError, AttributeError):
            _default_picker = None
    return _default_picker


def native_picker_available() -> bool:
    return default_picker() is not None


# Resolve the small packaged DLL while the extension module graph is loading,
# not on the first pointer event visible to the user.
default_picker()


__all__ = (
    "NativePickResult", "NativePointPicker", "default_library_path",
    "default_picker", "native_picker_available",
)
