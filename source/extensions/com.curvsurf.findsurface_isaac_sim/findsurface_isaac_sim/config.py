"""Schema-versioned, host-safe application defaults."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 3
PREVIOUS_SCHEMA_VERSION = 2
CONFIG_FILE_NAME = "app_defaults.json"
TRANSIENT_TOP_LEVEL_KEYS = frozenset({"capture", "cloud", "hover", "telemetry"})
FRONT_TO_UP_ORIENTATION_XYZW = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)

DEFAULT_APP_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "objects": {
        "plane": {
            "enabled": True,
            "subdivision": 96,
            "display_subdivision": 4,
            "face_color_rgba": [0.36, 0.72, 0.94, 0.25],
            "wire_color_rgba": [0.10, 0.43, 0.62, 1.0],
            "position_m": [-3.2, 0.0, 0.7],
            "orientation_xyzw": [-0.23167173145756478, -0.1838132326213639, 0.041640930695354204, 0.9543620577367707],
            "width_m": 1.2,
            "height_m": 1.2,
        },
        "sphere": {
            "enabled": True,
            "subdivision": 96,
            "display_subdivision": 32,
            "face_color_rgba": [0.36, 0.72, 0.94, 0.25],
            "wire_color_rgba": [0.10, 0.43, 0.62, 1.0],
            "position_m": [-1.6, 0.0, 0.55],
            "orientation_xyzw": list(FRONT_TO_UP_ORIENTATION_XYZW),
            "radius_m": 0.5,
        },
        "cylinder": {
            "enabled": True,
            "subdivision": 96,
            "display_subdivision": 32,
            "face_color_rgba": [0.36, 0.72, 0.94, 0.25],
            "wire_color_rgba": [0.10, 0.43, 0.62, 1.0],
            "position_m": [0.0, 0.0, 0.6],
            "orientation_xyzw": list(FRONT_TO_UP_ORIENTATION_XYZW),
            "radius_m": 0.45,
            "length_m": 1.2,
        },
        "cone": {
            "enabled": True,
            "subdivision": 96,
            "display_subdivision": 32,
            "face_color_rgba": [0.36, 0.72, 0.94, 0.25],
            "wire_color_rgba": [0.10, 0.43, 0.62, 1.0],
            "position_m": [1.6, 0.0, 0.6],
            "orientation_xyzw": list(FRONT_TO_UP_ORIENTATION_XYZW),
            "top_radius_m": 0.15,
            "bottom_radius_m": 0.5,
            "length_m": 1.2,
        },
        "torus": {
            "enabled": True,
            "subdivision": 96,
            "display_subdivision": 32,
            "face_color_rgba": [0.36, 0.72, 0.94, 0.25],
            "wire_color_rgba": [0.10, 0.43, 0.62, 1.0],
            "position_m": [3.2, 0.0, 0.65],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "major_radius_m": 0.45,
            "minor_radius_m": 0.15,
            "sweep_start_deg": 0.0,
            "sweep_angle_deg": 360.0,
        },
    },
    "sensor": {
        "source": "DepthSensor",
        "profile": "d455-16x9-50k",
        "enabled": True,
        "follows_camera": True,
        "disparity_noise_enabled": True,
        "disparity_noise_mean_px": 0.25,
        "disparity_noise_sigma_px": 0.25,
        "outlier_removal_enabled": True,
        "axial_error_enabled": False,
        "axial_error_sigma_m": 0.0015,
        "axial_error_model": "distance",
        "axial_error_slope_m_per_m": 0.0050,
        "axial_error_intercept_m": 0.0100,
        "target_material": {
            "base": "concrete",
            "coating": "paint",
            "attribute": "emissive",
        },
    },
    "cameras": {
        "scanner": {
            "projection": "perspective",
            "position_m": [0.0, 0.0, 8.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "scene": {
            "projection": "orthographic",
            "orthographic_height_m": 10.0,
            "position_m": [5.395475210886386, 5.395475210886386, 5.970475210886386],
            "orientation_xyzw": [-0.27984814233312133, 0.3647051996310009, 0.11591689595929515, 0.8804762392171493],
        },
    },
    "renderer": {
        "scene_view": "OpenGL",
        "scanner_view": "OpenGL",
        "point_depth_offset_m": 0.006,
    },
    "display": {
        "point_size": 2.5,
        "ruler_main_opacity": 0.8,
        "ruler_pip_opacity": 0.35,
        "inlier_size": 2.5,
        "outlier_color_rgba": [0.70, 0.78, 0.88, 1.0],
        "inlier_negative_rgba": [0.0, 0.22, 1.0, 1.0],
        "inlier_zero_rgba": [1.0, 1.0, 0.0, 1.0],
        "inlier_positive_rgba": [1.0, 0.02, 0.0, 1.0],
        "detected_preview_subdivision": 96,
        "detected_preview_face_rgba": [0.72, 0.96, 0.92, 0.25],
        "detected_preview_wire_rgba": [0.15, 1.0, 0.65, 1.0],
        "detected_mesh_appearances": {
            mode: {
                "subdivision": 4 if mode == "plane" else 32,
                "face_rgba": [0.72, 0.96, 0.92, 0.25],
                "wire_rgba": [0.15, 1.0, 0.65, 1.0],
            }
            for mode in ("plane", "sphere", "cylinder", "cone", "torus")
        },
        "captured_mesh_appearances": {
            mode: {
                "subdivision": 4 if mode == "plane" else 32,
                "face_rgba": [1.0, 0.72, 0.18, 0.22],
                "wire_rgba": [1.0, 0.78, 0.16, 1.0],
            }
            for mode in ("plane", "sphere", "cylinder", "cone", "torus")
        },
        "minimum_geometry_length_m": 0.01,
    },
    "picking": {
        "probe_radius_px": 6.0,
        "pick_nearest_on_probe_miss": True,
    },
    "preview": {
        "sticky_auto_preview": True,
        "sticky_auto_preview_timeout_ms": 500,
        "detection_average_window": 10,
    },
    "heatmap": {
        "surface_side_scale_mode": "sigma",
        "surface_side_fixed_scale_m": 0.01,
        "surface_side_sigma_multiplier": 3.0,
    },
    "detection": {
        "type": "Auto",
        "measurement_accuracy_m": 0.0505,
        "mean_distance_m": 0.1,
        "seed_radius_m": 1.025,
        "radial_expansion": 7,
        "lateral_extension": 5,
        "smart_convert_cone_to_cylinder": True,
        "smart_convert_torus_to_sphere": True,
        "smart_convert_torus_to_cylinder": True,
        "auto_preview": True,
    },
    "visibility": {"source": True, "detected": True, "captured": True, "points": True},
    "findsurface_ui": {
        "ranges": [[0.001, 0.1], [0.01, 0.2], [0.05, 2.0]],
    },
}


@dataclass(frozen=True)
class ConfigDiagnostic:
    code: str
    message: str
    path: Path


@dataclass(frozen=True)
class AppConfigLoadResult:
    config: dict[str, Any]
    path: Path
    created: bool = False
    used_fallback: bool = False
    diagnostics: tuple[ConfigDiagnostic, ...] = ()


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / CONFIG_FILE_NAME


def deep_merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(defaults))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def durable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return defaults-safe state, excluding transient runtime payloads."""

    durable = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in TRANSIENT_TOP_LEVEL_KEYS
    }
    durable.pop("theme", None)
    if isinstance(durable.get("renderer"), dict):
        durable["renderer"]["scene_view"] = "OpenGL"
    return durable


def load_app_config(path: os.PathLike[str] | str | None = None) -> AppConfigLoadResult:
    destination = Path(path) if path is not None else default_config_path()
    if not destination.exists():
        try:
            write_app_config(destination, DEFAULT_APP_CONFIG)
        except OSError as error:
            return _fallback(destination, "config-create-failed", error)
        return AppConfigLoadResult(
            config=copy.deepcopy(DEFAULT_APP_CONFIG), path=destination, created=True
        )

    try:
        loaded = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return _fallback(destination, "config-read-failed", error)
    if not isinstance(loaded, dict):
        return _fallback(destination, "config-root-invalid", "root must be a JSON object")
    loaded_version = loaded.get("schema_version", SCHEMA_VERSION)
    if loaded_version == PREVIOUS_SCHEMA_VERSION:
        loaded = _migrate_schema_two(loaded)
    elif loaded_version != SCHEMA_VERSION:
        return _fallback(
            destination,
            "config-schema-unsupported",
            f"expected schema_version {SCHEMA_VERSION}, got {loaded_version!r}",
        )
    from .axial_error import migrate_distance_error_sensor
    if isinstance(loaded.get("sensor"), dict):
        try:
            migrate_distance_error_sensor(
                loaded["sensor"], DEFAULT_APP_CONFIG["sensor"]["axial_error_slope_m_per_m"])
        except (TypeError, ValueError, OverflowError) as error:
            return _fallback(destination, "config-axial-error-invalid", error)
    merged = deep_merge(DEFAULT_APP_CONFIG, loaded)
    merged.pop("theme", None)
    merged["schema_version"] = SCHEMA_VERSION
    if isinstance(merged.get("renderer"), dict):
        merged["renderer"]["scene_view"] = "OpenGL"
    return AppConfigLoadResult(config=merged, path=destination)


def _migrate_schema_two(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the v2 torus start/end contract to v3 start/sweep in memory."""

    migrated = copy.deepcopy(dict(config))
    objects = migrated.get("objects")
    torus = objects.get("torus") if isinstance(objects, Mapping) else None
    if isinstance(torus, dict) and "sweep_angle_deg" not in torus:
        start = float(torus.get("sweep_start_deg", 0.0))
        end = float(torus.pop("sweep_end_deg", start + 360.0))
        torus["sweep_angle_deg"] = end - start
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


def write_app_config(path: os.PathLike[str] | str, config: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = durable_config(config)
    normalized["schema_version"] = SCHEMA_VERSION
    text = json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"
    destination.write_text(text, encoding="utf-8", newline="\n")


def _fallback(path: Path, code: str, error: object) -> AppConfigLoadResult:
    return AppConfigLoadResult(
        config=copy.deepcopy(DEFAULT_APP_CONFIG),
        path=path,
        used_fallback=True,
        diagnostics=(ConfigDiagnostic(code, str(error), path),),
    )


__all__ = [
    "AppConfigLoadResult",
    "CONFIG_FILE_NAME",
    "ConfigDiagnostic",
    "DEFAULT_APP_CONFIG",
    "FRONT_TO_UP_ORIENTATION_XYZW",
    "SCHEMA_VERSION",
    "TRANSIENT_TOP_LEVEL_KEYS",
    "deep_merge",
    "default_config_path",
    "durable_config",
    "load_app_config",
    "write_app_config",
]
