"""Stable, host-safe boundary around the imported FindSurface snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Callable
from collections.abc import Mapping

import numpy as np

from .domain import DetectionOptions, DetectionRequest, DetectionResult, GeometryMode


class FindSurfaceUnavailableError(RuntimeError):
    """FindSurface core or its native SDK cannot be used in this environment."""


_SMART_CONVERSION_KEYS = (
    "smart_convert_cone_to_cylinder",
    "smart_convert_torus_to_sphere",
    "smart_convert_torus_to_cylinder",
)


class _TimedFindSurfaceSdk:
    """Time only SDK find(), excluding cloud setup and result conversion."""

    def __init__(self, sdk):
        self._sdk = sdk
        self.last_call_duration_s = None
        self.last_call_started_s = None
        self.last_call_finished_s = None

    def __getattr__(self, name):
        return getattr(self._sdk, name)

    def find(self, *args, **kwargs):
        self.last_call_duration_s = None
        self.last_call_started_s = perf_counter()
        self.last_call_finished_s = None
        try:
            return self._sdk.find(*args, **kwargs)
        finally:
            self.last_call_finished_s = perf_counter()
            self.last_call_duration_s = self.last_call_finished_s - self.last_call_started_s


class _ConfiguredCoreDetector:
    """Strip app-only options and atomically configure the native provider."""

    def __init__(self, detector, provider) -> None:
        self._detector = detector
        self._provider = provider
        self._lock = Lock()

    @property
    def native_call_duration_s(self):
        sdk = getattr(self._provider, "_sdk", None)
        return sdk.last_call_duration_s if isinstance(sdk, _TimedFindSurfaceSdk) else None

    def detect(self, points, seed_index, options=None, active_mask=None, active_generation=None):
        native_options = options
        smart_conversions = (True, True, True)
        if isinstance(options, dict):
            native_options = dict(options)
            smart_conversions = tuple(
                bool(native_options.pop(name, True)) for name in _SMART_CONVERSION_KEYS
            )
        with self._lock:
            sdk = getattr(self._provider, "_sdk", None)
            if isinstance(sdk, _TimedFindSurfaceSdk):
                sdk.last_call_duration_s = None
                sdk.last_call_started_s = None
                sdk.last_call_finished_s = None
            configure = getattr(self._provider, "set_smart_conversions", None)
            if configure is not None:
                configure(*smart_conversions)
            return self._detector.detect(
                points,
                seed_index=seed_index,
                options=native_options,
                active_mask=active_mask,
                active_generation=active_generation,
            )

    def close(self) -> None:
        with self._lock:
            self._detector.close()


def _configurable_native_provider(native, sdk):
    """Adapt the immutable demo-1 provider without modifying its imported snapshot."""

    class ConfigurableNativeProvider(native.NativeFindSurfaceProvider):
        def __init__(self):
            self._smart_conversions = (True, True, True)
            super().__init__(sdk=sdk)

        def set_smart_conversions(
            self,
            cone_to_cylinder: bool,
            torus_to_sphere: bool,
            torus_to_cylinder: bool,
        ) -> None:
            self._smart_conversions = (
                bool(cone_to_cylinder),
                bool(torus_to_sphere),
                bool(torus_to_cylinder),
            )

        def _apply_options(self, context, options, active_seed_index):
            super()._apply_options(context, options, active_seed_index)
            for parameter, enabled in zip(
                native.SMART_CONVERSION_PARAMETERS, self._smart_conversions,
            ):
                self._sdk.set_boolean(context, parameter, enabled)

    return ConfigurableNativeProvider()


@dataclass(frozen=True)
class FindSurfaceAvailability:
    core_importable: bool
    native_library_path: str | None
    native_library_present: bool
    error: str | None = None


def probe_findsurface(
    *,
    library_path: str | Path | None = None,
    importer: Callable[[str], object] = import_module,
) -> FindSurfaceAvailability:
    """Inspect the core and native path without loading or downloading a library."""

    try:
        importer("findsurface.bootstrap")
        wrapper = importer("findsurface.bootstrap.findsurface_sdk.wrapper")
        resolved = Path(library_path) if library_path is not None else wrapper.default_library_path()
    except Exception as exception:
        return FindSurfaceAvailability(
            core_importable=False,
            native_library_path=str(library_path) if library_path is not None else None,
            native_library_present=False,
            error=f"{type(exception).__name__}: {exception}",
        )
    return FindSurfaceAvailability(
        core_importable=True,
        native_library_path=str(resolved),
        native_library_present=resolved.is_file(),
    )


def create_detector(
    *,
    provider=None,
    library_path: str | Path | None = None,
    allow_sdk_download: bool = False,
    importer: Callable[[str], object] = import_module,
):
    """Create the public detector while keeping native acquisition explicit."""

    try:
        core = importer("findsurface.bootstrap")
    except Exception as exception:
        raise FindSurfaceUnavailableError(
            f"FindSurface core import failed: {type(exception).__name__}: {exception}"
        ) from exception
    if provider is not None and library_path is not None:
        raise ValueError("provider and library_path are mutually exclusive")
    if provider is not None:
        return _ConfiguredCoreDetector(core.FindSurfaceDetector(provider=provider), provider)
    availability = probe_findsurface(library_path=library_path, importer=importer)
    if not availability.core_importable:
        raise FindSurfaceUnavailableError(availability.error or "FindSurface core is unavailable")
    if not availability.native_library_present and not allow_sdk_download:
        raise FindSurfaceUnavailableError(
            "FindSurface native SDK is unavailable. Install the release asset explicitly "
            f"at {availability.native_library_path}; automatic download is disabled by the app adapter."
        )
    try:
        native = importer("findsurface.bootstrap.native")
        wrapper = importer("findsurface.bootstrap.findsurface_sdk.wrapper")
        sdk = (
            wrapper.FindSurfaceSdk()
            if library_path is None
            else wrapper.FindSurfaceSdk(library_path=Path(library_path))
        )
        provider = _configurable_native_provider(native, _TimedFindSurfaceSdk(sdk))
    except Exception as exception:
        raise FindSurfaceUnavailableError(
            f"FindSurface native SDK initialization failed: {type(exception).__name__}: {exception}"
        ) from exception
    return _ConfiguredCoreDetector(core.FindSurfaceDetector(provider=provider), provider)


def _upstream_options(options: DetectionOptions) -> dict[str, object]:
    return {
        "detection_type": options.geometry_mode.value,
        "seed_radius": options.seed_radius_m,
        "measurement_accuracy": options.measurement_accuracy_m,
        "mean_distance": options.mean_distance_m,
        "radial_expansion": options.radial_expansion,
        "lateral_extension": options.lateral_extension,
        "smart_convert_cone_to_cylinder": options.smart_convert_cone_to_cylinder,
        "smart_convert_torus_to_sphere": options.smart_convert_torus_to_sphere,
        "smart_convert_torus_to_cylinder": options.smart_convert_torus_to_cylinder,
    }


def _plain_result_numbers(value):
    """Normalize scalar types only; preserve the SDK's coordinates and values.

    Demo-1 partial torus frame math can propagate np.float32 from inlier
    coordinates. The strict domain freezer intentionally accepts Python values.
    Do not apply this conversion to the input point buffer or accept arrays here.
    """
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _plain_result_numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain_result_numbers(item) for item in value)
    return value


class DomainFindSurfaceDetector:
    """Map immutable app requests to the imported detector without changing coordinates."""

    def __init__(self, detector) -> None:
        self._detector = detector
        self._closed = False
        self._mask_frame = None
        self._active_mask: tuple[bool, ...] = ()
        self._native_call_duration_s = None

    @property
    def native_call_duration_s(self):
        return self._native_call_duration_s

    def detect(self, request: DetectionRequest) -> DetectionResult:
        self._native_call_duration_s = None
        if self._closed:
            raise RuntimeError("FindSurface domain detector is closed")
        if request.frame is not self._mask_frame:
            self._mask_frame = request.frame
            self._active_mask = (True,) * request.frame.compacted_count
        native_options = _upstream_options(request.options)
        try:
            native_result = self._detector.detect(
                request.frame.points_world_m,
                seed_index=request.selection.seed_index,
                options=native_options,
                active_mask=self._active_mask,
                active_generation=request.frame.sequence,
            )
        finally:
            self._native_call_duration_s = getattr(self._detector, "native_call_duration_s", None)
        try:
            geometry_type = GeometryMode(str(native_result.type))
        except ValueError as exception:
            raise ValueError(f"unsupported FindSurface result type: {native_result.type!r}") from exception
        if geometry_type is GeometryMode.AUTO:
            raise ValueError("FindSurface result cannot have Auto geometry type")
        indices = tuple(int(index) for index in native_result.inlier_indices)
        if any(index >= request.frame.compacted_count for index in indices):
            raise ValueError("FindSurface returned an out-of-range inlier index")
        parameters = _plain_result_numbers(native_result.parameters)
        return DetectionResult(
            request_key=request.key,
            generations=request.generations,
            geometry_type=geometry_type,
            parameters=parameters,
            inlier_indices=indices,
            rms_error_m=float(native_result.rms_error),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mask_frame = None
        self._active_mask = ()
        self._detector.close()


def create_domain_detector(**kwargs) -> DomainFindSurfaceDetector:
    return DomainFindSurfaceDetector(create_detector(**kwargs))


__all__ = [
    "FindSurfaceAvailability",
    "DomainFindSurfaceDetector",
    "FindSurfaceUnavailableError",
    "create_domain_detector",
    "create_detector",
    "probe_findsurface",
]
