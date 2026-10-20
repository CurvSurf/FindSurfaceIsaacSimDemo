"""Versioned, host-safe persistence for durable user choices.

The package ``app_defaults.json`` remains the reset/default source.  This module
stores only the user-editable preferences that are safe to carry between app
sessions; point clouds, captures, and sensor run
mode deliberately stay transient.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .command_facade import SessionSnapshot, session_from_config
from .config import deep_merge, DEFAULT_APP_CONFIG
from .depth_sensor import DEFAULT_DEPTH_PROFILE_ID, depth_profile
from .domain import SensorKind
from .lidar_sensor import lidar_profile, SCANNER_LIDAR_PROFILES
from .sensor_diagnostics import DepthSensorDiagnostics
from .axial_error import migrate_distance_error_sensor


USER_PREFERENCES_SCHEMA_VERSION = 1
PERSISTENT_SETTING_PATH = (
    "/persistent/exts/com.curvsurf.findsurface_isaac_sim/userPreferences"
)

_ALLOWED_FIELDS: dict[str, object] = {
    "cameras": {
        role: {"projection": str, "position_m": list, "orientation_xyzw": list,
               "target_world_m": list, "vertical_fov_degrees": (int, float),
               "near_m": (int, float), "far_m": (int, float),
               "orthographic_height_m": (int, float), "manual_fov": bool}
        for role in ("scanner", "scene")
    },
    "findsurface_ui": {"ranges": list},
    "objects": {
        mode: {key: (bool if isinstance(value, bool) else list if isinstance(value, list) else (int, float))
               for key, value in fields.items()}
        for mode, fields in DEFAULT_APP_CONFIG["objects"].items()
    },
    "sensor": {
        "source": str,
        "profile": str,
        "disparity_noise_enabled": bool,
        "disparity_noise_mean_px": (int, float),
        "disparity_noise_sigma_px": (int, float),
        "outlier_removal_enabled": bool,
        "axial_error_enabled": bool,
        "axial_error_sigma_m": (int, float),
        "axial_error_model": str,
        "axial_error_slope_m_per_m": (int, float),
        "axial_error_intercept_m": (int, float),
        "axial_error_sigma_at_1m_m": (int, float),
        "target_material": {"base": str, "coating": str, "attribute": str},
    },
    "detection": {
        "type": str,
        "measurement_accuracy_m": (int, float),
        "mean_distance_m": (int, float),
        "seed_radius_m": (int, float),
        "radial_expansion": int,
        "lateral_extension": int,
        "smart_convert_cone_to_cylinder": bool,
        "smart_convert_torus_to_sphere": bool,
        "smart_convert_torus_to_cylinder": bool,
    },
    "display": {
        "point_size": (int, float),
        "ruler_main_opacity": (int, float),
        "ruler_pip_opacity": (int, float),
        "inlier_size": (int, float),
        "outlier_color_rgba": list,
        "inlier_negative_rgba": list,
        "inlier_zero_rgba": list,
        "inlier_positive_rgba": list,
        "detected_preview_subdivision": int,
        "detected_preview_face_rgba": list,
        "detected_preview_wire_rgba": list,
        "detected_mesh_appearances": {
            mode: {"subdivision": int, "face_rgba": list, "wire_rgba": list}
            for mode in ("plane", "sphere", "cylinder", "cone", "torus")
        },
        "captured_mesh_appearances": {
            mode: {"subdivision": int, "face_rgba": list, "wire_rgba": list}
            for mode in ("plane", "sphere", "cylinder", "cone", "torus")
        },
        "minimum_geometry_length_m": (int, float),
    },
    "picking": {
        "probe_radius_px": (int, float),
        "pick_nearest_on_probe_miss": bool,
    },
    "preview": {
        "sticky_auto_preview": bool,
        "sticky_auto_preview_timeout_ms": int,
        "detection_average_window": int,
    },
    "heatmap": {
        "surface_side_scale_mode": str,
        "surface_side_fixed_scale_m": (int, float),
        "surface_side_sigma_multiplier": (int, float),
    },
}


@dataclass(frozen=True)
class UserPreferencesDiagnostic:
    code: str
    message: str


@dataclass(frozen=True)
class UserPreferencesLoadResult:
    config: dict[str, Any]
    loaded: bool = False
    diagnostic: UserPreferencesDiagnostic | None = None


def depth_diagnostics_from_config(config: Mapping[str, object]) -> DepthSensorDiagnostics:
    sensor = dict(config["sensor"])
    migrate_distance_error_sensor(sensor)
    return DepthSensorDiagnostics(
        disparity_noise_enabled=bool(sensor["disparity_noise_enabled"]),
        disparity_noise_mean_px=float(sensor["disparity_noise_mean_px"]),
        disparity_noise_sigma_px=float(sensor["disparity_noise_sigma_px"]),
        outlier_removal_enabled=bool(sensor["outlier_removal_enabled"]),
        axial_error_enabled=bool(sensor["axial_error_enabled"]),
        axial_error_sigma_m=float(sensor["axial_error_sigma_m"]),
        axial_error_model=sensor.get("axial_error_model", "constant"),
        axial_error_slope_m_per_m=float(sensor.get("axial_error_slope_m_per_m", 0.0)),
        axial_error_intercept_m=float(sensor.get("axial_error_intercept_m", sensor["axial_error_sigma_m"])),
    )


def encode_user_preferences(
    snapshot: SessionSnapshot, diagnostics: DepthSensorDiagnostics,
) -> str:
    """Serialize only durable preferences from the current immutable state."""

    sensor = snapshot.sensor
    detection = snapshot.detection
    interaction = snapshot.interaction
    payload = {
        "schema_version": USER_PREFERENCES_SCHEMA_VERSION,
        "cameras": {
            view.role.value: {
                "projection": "orthographic" if view.intrinsics.orthographic_height_m is not None else "perspective",
                "position_m": view.pose_world.position_m,
                "orientation_xyzw": view.pose_world.orientation_xyzw,
                "target_world_m": view.target_world_m,
                "vertical_fov_degrees": view.intrinsics.vertical_fov_degrees,
                "near_m": view.intrinsics.near_m,
                "far_m": view.intrinsics.far_m,
                "orthographic_height_m": view.intrinsics.orthographic_height_m or 10.0,
                "manual_fov": snapshot.scanner_manual_fov if view.role.value == "scanner" else True,
            } for view in snapshot.camera_views
        },
        "objects": {item.geometry_type.value.lower(): item.to_config() for item in snapshot.objects},
        "findsurface_ui": {"ranges": interaction.fs_parameter_ranges},
        "sensor": {
            "source": "DepthSensor" if sensor.sensor_kind is SensorKind.DEPTH else "RTX",
            "profile": sensor.profile_id,
            "disparity_noise_enabled": diagnostics.disparity_noise_enabled,
            "disparity_noise_mean_px": diagnostics.disparity_noise_mean_px,
            "disparity_noise_sigma_px": diagnostics.disparity_noise_sigma_px,
            "outlier_removal_enabled": diagnostics.outlier_removal_enabled,
            "axial_error_enabled": diagnostics.axial_error_enabled,
            "axial_error_sigma_m": diagnostics.axial_error_sigma_m,
            "axial_error_model": diagnostics.axial_error_model,
            "axial_error_slope_m_per_m": diagnostics.axial_error_slope_m_per_m,
            "axial_error_intercept_m": diagnostics.axial_error_intercept_m,
            "target_material": {
                "base": sensor.target_material.base,
                "coating": sensor.target_material.coating,
                "attribute": sensor.target_material.attribute,
            },
        },
        "detection": {
            "type": detection.geometry_mode.value,
            "measurement_accuracy_m": detection.measurement_accuracy_m,
            "mean_distance_m": detection.mean_distance_m,
            "seed_radius_m": detection.seed_radius_m,
            "radial_expansion": detection.radial_expansion,
            "lateral_extension": detection.lateral_extension,
            "smart_convert_cone_to_cylinder": detection.smart_convert_cone_to_cylinder,
            "smart_convert_torus_to_sphere": detection.smart_convert_torus_to_sphere,
            "smart_convert_torus_to_cylinder": detection.smart_convert_torus_to_cylinder,
        },
        "display": {
            "point_size": interaction.point_rendering_size,
            "ruler_main_opacity": interaction.ruler_main_opacity,
            "ruler_pip_opacity": interaction.ruler_pip_opacity,
            "inlier_size": interaction.inlier_rendering_size,
            "outlier_color_rgba": interaction.outlier_color_rgba,
            "inlier_negative_rgba": interaction.inlier_negative_rgba,
            "inlier_zero_rgba": interaction.inlier_zero_rgba,
            "inlier_positive_rgba": interaction.inlier_positive_rgba,
            "detected_preview_subdivision": interaction.detected_preview_subdivision,
            "detected_preview_face_rgba": interaction.detected_preview_face_rgba,
            "detected_preview_wire_rgba": interaction.detected_preview_wire_rgba,
            "detected_mesh_appearances": {
                mode.value.lower(): {
                    "subdivision": appearance.subdivision,
                    "face_rgba": appearance.face_rgba,
                    "wire_rgba": appearance.wire_rgba,
                }
                for mode, appearance in interaction.detected_mesh_appearances
            },
            "captured_mesh_appearances": {
                mode.value.lower(): {
                    "subdivision": appearance.subdivision,
                    "face_rgba": appearance.face_rgba,
                    "wire_rgba": appearance.wire_rgba,
                }
                for mode, appearance in interaction.captured_mesh_appearances
            },
            "minimum_geometry_length_m": interaction.minimum_geometry_length_m,
        },
        "picking": {
            "probe_radius_px": interaction.probe_radius_px,
            "pick_nearest_on_probe_miss": interaction.pick_nearest_on_probe_miss,
        },
        "preview": {
            "sticky_auto_preview": interaction.sticky_auto_preview,
            "sticky_auto_preview_timeout_ms": interaction.sticky_preview_timeout_ms,
            "detection_average_window": interaction.detection_average_window,
        },
        "heatmap": {
            "surface_side_scale_mode": interaction.surface_side_scale_mode.value,
            "surface_side_fixed_scale_m": interaction.surface_side_fixed_scale_m,
            "surface_side_sigma_multiplier": interaction.surface_side_sigma_multiplier,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def load_user_preferences(
    defaults: Mapping[str, object], serialized: str | None,
) -> UserPreferencesLoadResult:
    """Apply a validated preference blob over app defaults without mutating them."""

    base = copy.deepcopy(dict(defaults))
    if isinstance(base.get("sensor"), dict):
        migrate_distance_error_sensor(base["sensor"])
    _normalize_sensor_selection(base)
    if not serialized:
        return UserPreferencesLoadResult(base)
    try:
        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise ValueError("preference root must be a JSON object")
        version = payload.get("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != USER_PREFERENCES_SCHEMA_VERSION
        ):
            raise ValueError(
                f"expected schema_version {USER_PREFERENCES_SCHEMA_VERSION}, got {version!r}"
            )
        # Retired ruler-only units must not invalidate the remaining saved settings.
        if isinstance(payload.get("display"), dict):
            payload["display"].pop("ruler_unit", None)
        overrides = _validated_overrides(payload)
        sensor_override = overrides.get("sensor", {})
        migrate_distance_error_sensor(sensor_override, base["sensor"]["axial_error_slope_m_per_m"])
        if "axial_error_sigma_m" in sensor_override and "axial_error_model" not in sensor_override:
            # Pre-model preferences describe constant noise, not the new default model.
            sensor_override["axial_error_model"] = "constant"
        display_override = overrides.get("display", {})
        legacy_detected = {
            "subdivision": display_override.get("detected_preview_subdivision"),
            "face_rgba": display_override.get("detected_preview_face_rgba"),
            "wire_rgba": display_override.get("detected_preview_wire_rgba"),
        }
        if (
            "detected_mesh_appearances" not in display_override
            and any(value is not None for value in legacy_detected.values())
        ):
            defaults_display = base["display"]
            fallback = {
                "subdivision": defaults_display["detected_preview_subdivision"],
                "face_rgba": defaults_display["detected_preview_face_rgba"],
                "wire_rgba": defaults_display["detected_preview_wire_rgba"],
            }
            display_override["detected_mesh_appearances"] = {
                mode: {
                    name: legacy_detected[name]
                    if legacy_detected[name] is not None else fallback[name]
                    for name in fallback
                }
                for mode in ("plane", "sphere", "cylinder", "cone", "torus")
            }
        candidate = deep_merge(base, overrides)
        _normalize_sensor_selection(candidate)
        snapshot = session_from_config(candidate)
        depth_diagnostics_from_config(candidate)
        profile_id = snapshot.sensor.profile_id
        if profile_id != "default":
            (depth_profile if snapshot.sensor.sensor_kind is SensorKind.DEPTH else lidar_profile)(profile_id)
    except (KeyError, OverflowError, TypeError, ValueError, json.JSONDecodeError) as error:
        return UserPreferencesLoadResult(
            base,
            diagnostic=UserPreferencesDiagnostic("user-preferences-invalid", str(error)),
        )
    return UserPreferencesLoadResult(candidate, loaded=True)


def _normalize_sensor_selection(config: dict[str, object]) -> None:
    """Keep supported test LiDAR choices; migrate retired profiles to D455."""

    sensor = config.get("sensor")
    if not isinstance(sensor, dict):
        return
    if sensor.get("source") == "RTX" and (
        sensor.get("profile") == "default" or sensor.get("profile") in {
            profile.profile_id for profile in SCANNER_LIDAR_PROFILES
        }
    ):
        return
    if sensor.get("source") != "DepthSensor":
        sensor["source"] = "DepthSensor"
        sensor["profile"] = DEFAULT_DEPTH_PROFILE_ID


def _validated_overrides(payload: Mapping[str, object]) -> dict[str, object]:
    unknown = set(payload) - {"schema_version", *_ALLOWED_FIELDS}
    if unknown:
        raise ValueError(f"unknown preference fields: {', '.join(sorted(unknown))}")
    result: dict[str, object] = {}
    for section, schema in _ALLOWED_FIELDS.items():
        if section not in payload:
            continue
        value = payload[section]
        if not isinstance(value, Mapping):
            raise TypeError(f"{section} must be an object")
        result[section] = _validate_mapping(section, value, schema)
    return result


def _validate_mapping(
    path: str, values: Mapping[str, object], schema: Mapping[str, object],
) -> dict[str, object]:
    unknown = set(values) - set(schema)
    if unknown:
        raise ValueError(f"unknown fields at {path}: {', '.join(sorted(unknown))}")
    result: dict[str, object] = {}
    for name, value in values.items():
        expected = schema[name]
        field_path = f"{path}.{name}"
        if isinstance(expected, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError(f"{field_path} must be an object")
            result[name] = _validate_mapping(field_path, value, expected)
            continue
        if isinstance(value, bool) and expected is not bool:
            raise TypeError(f"{field_path} has invalid boolean value")
        if not isinstance(value, expected):
            raise TypeError(f"{field_path} has invalid type {type(value).__name__}")
        result[name] = value
    return result


__all__ = [
    "PERSISTENT_SETTING_PATH",
    "USER_PREFERENCES_SCHEMA_VERSION",
    "UserPreferencesDiagnostic",
    "UserPreferencesLoadResult",
    "depth_diagnostics_from_config",
    "encode_user_preferences",
    "load_user_preferences",
]
