"""Isaac Sim 6.0.1 RTX LiDAR profile and complete-scan runtime adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import importlib.util
import math
import time
from typing import Any, Awaitable, Callable, Protocol, Sequence

import numpy as np

from .domain import Pose, ScanFrame, SensorKind, SensorLifecycle, SensorProfileDescriptor, SensorStatus, StageMetadata
from .scan import NativePointPayload, ScanDispatcher
from .scene import StagePose, StageSpace
from .sensor_material import SensorMaterialSetting, recommended_lidar_material
from .sensor_diagnostics import DepthSensorDiagnostics
from .sensors import ProviderTelemetry, SensorOperationToken, SensorProviderStateMachine


LIDAR_EXTENSION_VERSION = "1.4.6"
LIDAR_CONFIG = "OS1"
LIDAR_ASSET_URI = "/Isaac/Sensors/Ouster/OS1/OS1.usd"
LIDAR_ANNOTATOR = "generic-model-output"
DEFAULT_LIDAR_PROFILE_ID = "os0-rev6-128ch-10hz-512"
OFFICIAL_NON_NVIDIA_CONFIG_COUNT = 17
OFFICIAL_NON_NVIDIA_PROFILE_COUNT = 163
SCANNER_LIDAR_PROFILE_COUNT = 10
SCANNER_EXCLUDED_PROFILE_COUNT = 153

# Product cameras use local -Z forward, +X right, +Y up while official
# OmniLidar assets use +X forward, +Y left, +Z up.  This fixed child mount rotates the
# physical LiDAR rays into the scanner-camera frame; it is not a point-cloud
# post-process.  Quaternion component order is x, y, z, w.
LIDAR_TO_CAMERA_MOUNT_ORIENTATION_XYZW = (-0.5, 0.5, 0.5, 0.5)


@dataclass(frozen=True)
class LidarProfile:
    profile_id: str
    label: str
    variant: str | tuple[tuple[str, str], ...] | None
    channels: int
    horizontal_resolution: int
    scan_rate_hz: float
    config: str = LIDAR_CONFIG
    asset_uri: str = LIDAR_ASSET_URI
    manufacturer: str = "Ouster"
    sensor_model: str = "OS1 REV6"
    scan_type: str = "rotary_3d"
    function: str = "General-purpose 360-degree 3D ranging and mapping."
    features: str = "Mid-range spinning LiDAR with accumulated full scans."
    profile_summary: str = ""
    minimum_range_m: float = 0.3
    maximum_range_m: float = 120.0
    nominal_ray_count: int | None = None
    certified: bool = False
    range_summary: str = ""

    def __post_init__(self) -> None:
        if not all((self.profile_id, self.label, self.config, self.asset_uri, self.manufacturer, self.sensor_model)):
            raise ValueError("LiDAR profile identity strings cannot be empty")
        variant = self.variant
        if isinstance(variant, tuple):
            if not variant or any(not key or not value for key, value in variant):
                raise ValueError("nested LiDAR variant entries cannot be empty")
        elif variant is not None and not variant:
            raise ValueError("LiDAR variant cannot be empty")
        if self.channels <= 0 or self.horizontal_resolution <= 0:
            raise ValueError("LiDAR ray dimensions must be positive")
        if not math.isfinite(self.scan_rate_hz) or self.scan_rate_hz <= 0.0:
            raise ValueError("scan_rate_hz must be finite and positive")
        if self.scan_type not in {"rotary_2d", "rotary_3d", "solid_state"}:
            raise ValueError("scan_type must be rotary_2d, rotary_3d, or solid_state")
        if (
            not math.isfinite(self.minimum_range_m)
            or not math.isfinite(self.maximum_range_m)
            or self.minimum_range_m < 0.0
            or self.maximum_range_m <= self.minimum_range_m
        ):
            raise ValueError("LiDAR range bounds must be finite and increasing")
        if self.nominal_ray_count is not None and self.nominal_ray_count <= 0:
            raise ValueError("nominal_ray_count must be positive when provided")

    @property
    def emitted_rays_per_scan(self) -> int:
        return self.nominal_ray_count or self.channels * self.horizontal_resolution

    @property
    def is_single_scan_line(self) -> bool:
        """Return whether the native profile emits only one vertical scan line."""

        return self.channels == 1

    @property
    def variant_argument(self) -> str | dict[str, str] | None:
        if isinstance(self.variant, tuple):
            return dict(self.variant)
        return self.variant

    @property
    def scan_type_label(self) -> str:
        return {
            "rotary_2d": "Rotary 2D",
            "rotary_3d": "Rotary 3D",
            "solid_state": "Solid State",
        }[self.scan_type]

    @property
    def description_rows(self) -> tuple[tuple[str, str], ...]:
        certification = " · manufacturer-certified" if self.certified else ""
        profile = self.profile_summary or (
            f"{self.channels} channel(s) · {self.horizontal_resolution} samples · "
            f"{self.scan_rate_hz:g} Hz"
        )
        return (
            ("Name", f"{self.manufacturer} {self.sensor_model}"),
            ("Type", f"{self.scan_type_label}{certification}"),
            ("Function", self.function),
            ("Profile", profile),
            ("Range", self.range_summary or f"{self.minimum_range_m:g} to {self.maximum_range_m:g} m"),
            ("Features", self.features),
        )

    @property
    def recommended_material(self) -> SensorMaterialSetting:
        """Return the app-defined high-return target used when this profile is selected."""

        return recommended_lidar_material(self.manufacturer)

    @property
    def descriptor(self) -> SensorProfileDescriptor:
        return SensorProfileDescriptor(
            self.profile_id,
            SensorKind.RTX_LIDAR,
            model={
                "manufacturer": self.manufacturer,
                "sensor_model": self.sensor_model,
                "config": self.config,
                "variant": self.variant_argument,
                "scan_type": self.scan_type,
                "channels": self.channels,
                "horizontal_resolution": self.horizontal_resolution,
                "emitted_rays_per_scan": self.emitted_rays_per_scan,
                "scan_rate_hz": self.scan_rate_hz,
                "minimum_range_m": self.minimum_range_m,
                "maximum_range_m": self.maximum_range_m,
                "certified": self.certified,
                "accumulate_outputs": True,
                "annotator": LIDAR_ANNOTATOR,
            },
            calibration_provenance={
                "runtime": "Isaac Sim 6.0.1",
                "extension": f"isaacsim.sensors.experimental.rtx {LIDAR_EXTENSION_VERSION}",
                "asset_uri": self.asset_uri,
                "output_frame_of_reference": "SENSOR",
                "native_spherical_units": "azimuth/elevation degrees, range metres",
                "sensor_axes": "+X forward, +Y left, +Z up",
                "scanner_camera_axes": "+X right, +Y up, -Z forward",
                "physical_mount_orientation_xyzw": LIDAR_TO_CAMERA_MOUNT_ORIENTATION_XYZW,
                "physical_mount_axis_mapping": "+X->-Z, +Y->-X, +Z->+Y",
                "application_axial_error_configurable": True,
            },
        )


def _slug(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "p").replace(" ", "-")


def _ouster_profiles() -> tuple[LidarProfile, ...]:
    result = []
    models = (
        ("OS0", "/Isaac/Sensors/Ouster/OS0/OS0.usd", (("REV6", (128,), 0.3, 50.0), ("REV7", (128,), 0.5, 100.0)),
         "Ultra-wide short-range 360-degree mapping and obstacle perception."),
        ("OS1", "/Isaac/Sensors/Ouster/OS1/OS1.usd", (("REV6", (32, 128), 0.3, 120.0), ("REV7", (128,), 0.5, 170.0)),
         "General-purpose mid-range 360-degree mapping and perception."),
        ("OS2", "/Isaac/Sensors/Ouster/OS2/OS2.usd", (("REV6", (128,), 1.0, 240.0), ("REV7", (128,), 0.8, 350.0)),
         "Long-range 360-degree autonomous-vehicle and mapping perception."),
    )
    rates_and_resolutions = ((10, 512), (10, 1024), (10, 2048), (20, 512), (20, 1024))
    for model, asset_uri, revisions, function in models:
        for revision, channel_options, near_m, far_m in revisions:
            for channels in channel_options:
                for rate_hz, resolution in rates_and_resolutions:
                    variant = f"{model}_{revision}_{channels}ch{rate_hz}hz{resolution}res"
                    ray_count = channels * resolution
                    result.append(LidarProfile(
                        f"{model.lower()}-{revision.lower()}-{channels}ch-{rate_hz}hz-{resolution}",
                        f"Ouster {model} {revision} · {channels}ch {rate_hz}Hz {resolution}",
                        variant, channels, resolution, float(rate_hz),
                        config=model, asset_uri=asset_uri, manufacturer="Ouster",
                        sensor_model=f"{model} {revision}", scan_type="rotary_3d",
                        function=function,
                        features="Full 360° rotary scan; selectable channel count, scan rate, and azimuth resolution.",
                        profile_summary=f"{channels} channels · {resolution} azimuth samples · {ray_count:,} nominal rays/scan · {rate_hz} Hz",
                        minimum_range_m=near_m, maximum_range_m=far_m,
                        nominal_ray_count=ray_count,
                    ))
    return tuple(result)


def _decode_number(value: str) -> float:
    return float(value.replace("p", "."))


def _profile_token(profile: str, suffix: str) -> float | None:
    for token in profile.split("_"):
        if token.endswith(suffix):
            try:
                return _decode_number(token[:-len(suffix)])
            except ValueError:
                return None
    return None


def _sick_profile(
    *, config: str, asset_uri: str, model: str, profile: str,
    function: str, scan_type: str = "solid_state", product: str | None = None,
    fixed_rate_hz: float | None = None, fixed_resolution_deg: float | None = None,
    minimum_range_m: float = 0.0, maximum_range_m: float = 500.0,
    channels: int = 1,
) -> LidarProfile:
    rate = _profile_token(profile, "Hz") or fixed_rate_hz or 10.0
    resolution = _profile_token(profile, "deg") or fixed_resolution_deg
    range_in_name = _profile_token(profile, "m")
    if range_in_name is not None:
        maximum_range_m = range_in_name
    range_summary = (
        f"{minimum_range_m:g} to {maximum_range_m:g} m"
        if range_in_name is not None or maximum_range_m < 500.0
        else "Read from the selected official USD profile at runtime"
    )
    variant = (("Product", product), ("Profile", profile)) if product else (("Profile", profile),)
    selection = f"{product} · {profile}" if product else profile
    detail_parts = [f"{rate:g} Hz"]
    if resolution is not None:
        detail_parts.append(f"{resolution:g}° angular resolution")
    detail_parts.append("native USD ray pattern")
    return LidarProfile(
        f"{_slug(config)}-{_slug(product or '')}-{_slug(profile)}".replace("--", "-"),
        f"SICK {model} · {selection}", variant, channels, 1, rate,
        config=config, asset_uri=asset_uri, manufacturer="SICK", sensor_model=model,
        scan_type=scan_type, function=function,
        features="Manufacturer-certified industrial LiDAR; Product/Profile variants preserve the official USD selection.",
        profile_summary=" · ".join(detail_parts), minimum_range_m=minimum_range_m,
        maximum_range_m=maximum_range_m, certified=True, range_summary=range_summary,
    )


def _sick_profiles() -> tuple[LidarProfile, ...]:
    result = []
    lms4000_asset = "/Isaac/Sensors/SICK/LMS4000/SICK_LMS4000.usd"
    for product, profile, far_m in (
        ("LMS4111R", "Profile01_3p0m_0p0833deg", 3.0),
        ("LMS4121R", "Profile01_3p0m_0p0833deg", 3.0),
        ("LMS4124R", "Profile01_5p5m_0p0833deg", 5.5),
    ):
        result.append(_sick_profile(
            config="SICK_LMS4000", asset_uri=lms4000_asset, model="LMS4000",
            product=product, profile=profile, fixed_rate_hz=600.0,
            function="High-speed 2D contour measurement for material handling and intralogistics.",
            maximum_range_m=far_m,
        ))

    lms5xx_profiles = (
        "Profile00_25Hz_0p1667deg", "Profile01_25Hz_0p25deg", "Profile02_35Hz_0p25deg",
        "Profile03_35Hz_0p5deg", "Profile04_50Hz_0p3333deg", "Profile05_50Hz_0p5deg",
        "Profile06_75Hz_0p5deg", "Profile07_75Hz_1p0deg", "Profile08_100Hz_0p667deg",
        "Profile09_100Hz_1p0deg", "Profile10_50Hz_0p1667deg", "Profile11_75Hz_0p25deg",
        "Profile12_100Hz_0p1667deg", "Profile13_100Hz_0p3333deg", "Profile14_100Hz_0p5deg",
        "Profile15_25Hz_0p083deg", "Profile16_25Hz_0p042deg",
    )
    lms5xx_products = (
        ("LMS5xxExtendedRange", lms5xx_profiles),
        ("LMS5xxLiteHR", tuple(lms5xx_profiles[index] for index in (1, 3, 5, 6, 7))),
        ("LMS5xxLiteSR", tuple(lms5xx_profiles[index] for index in (1, 3, 5, 6, 7))),
        ("LMS5xxProHR", lms5xx_profiles),
        ("LMS5xxProSR", lms5xx_profiles),
    )
    for product, profiles in lms5xx_products:
        for profile in profiles:
            result.append(_sick_profile(
                config="SICK_LMS5xx", asset_uri="/Isaac/Sensors/SICK/LMS5xx/SICK_LMS5xx.usd",
                model="LMS5xx", product=product, profile=profile,
                function="Industrial 2D scanning for outdoor ranging, collision avoidance, and localization.",
            ))

    for profile in (
        "Profile01_12p5Hz_0p04deg", "Profile02_12p5Hz_0p06deg", "Profile04_12p5Hz_0p1deg",
        "Profile05_12p5Hz_0p12deg", "Profile11_12p5Hz_0p02deg", "Profile31_12p5Hz_0p24deg",
        "Profile32_12p5Hz_0p12deg", "Profile33_12p5Hz_0p08deg", "Profile35_12p5Hz_0p04deg",
        "Profile61_25Hz_0p08deg", "Profile62_25Hz_0p12deg", "Profile64_25Hz_0p2deg",
        "Profile65_25Hz_0p24deg", "Profile71_25Hz_0p04deg", "Profile91_25Hz_0p48deg",
        "Profile92_25Hz_0p24deg", "Profile93_25Hz_0p16deg", "Profile95_25Hz_0p08deg",
    ):
        result.append(_sick_profile(
            config="SICK_LRS4000", asset_uri="/Isaac/Sensors/SICK/LRS4000/SICK_LRS4000.usd",
            model="LRS4000", profile=profile,
            function="Long-range outdoor 2D scanning and localization.",
        ))

    for profile in (
        "Profile01_4p0m_33p3Hz", "Profile02_4p0m_25p0Hz", "Profile03_5p5m_33p3Hz",
        "Profile04_5p5m_25p0Hz", "Profile05_9p0m_25p0Hz", "Profile06_9p0m_20p0Hz",
    ):
        result.append(_sick_profile(
            config="SICK_microScan3", asset_uri="/Isaac/Sensors/SICK/microScan3/SICK_microScan3.usd",
            model="microScan3", profile=profile,
            function="Safety 2D laser scanning for mobile-platform collision protection.",
        ))

    for profile in (
        "Profile01_12p5Hz_0p25deg", "Profile02_6p25Hz_0p125deg", "Profile03_3p125Hz_0p0625deg",
    ):
        result.append(_sick_profile(
            config="SICK_MRS1000", asset_uri="/Isaac/Sensors/SICK/MRS1000/SICK_MRS1000.usd",
            model="MRS1000", profile=profile, channels=4,
            function="Four-layer 3D obstacle detection for mobile, traffic, and security applications.",
        ))

    for product, profile in (
        ("multiScan136", "Profile01_20Hz_0p125deg"),
        ("multiScan165", "Profile01_20Hz_0p5deg"),
        ("multiScan165S", "Profile01_20Hz_0p5deg"),
        ("multiScan166", "Profile01_20Hz_0p125deg"),
    ):
        result.append(_sick_profile(
            config="SICK_multiScan100", asset_uri="/Isaac/Sensors/SICK/multiScan100/SICK_multiScan100.usd",
            model="multiScan100", product=product, profile=profile, channels=16,
            function="Multi-layer 3D obstacle and drop-off detection for mobile and stationary platforms.",
        ))

    pico_common = (
        "Profile01_15Hz_0p5deg", "Profile02_15Hz_0p33deg", "Profile03_20Hz_0p1deg",
        "Profile04_20Hz_0p25deg", "Profile05_25Hz_0p25deg", "Profile06_30Hz_0p1deg",
        "Profile07_40Hz_0p25deg", "Profile08_50Hz_0p25deg", "Profile09_15Hz_0p05deg",
        "Profile10_40Hz_0p125deg", "Profile11_15Hz_1p0deg",
    )
    for product, profiles in (
        ("picoScan150Pro", pico_common),
        ("picoScan150Prime", pico_common),
        ("picoScan150Core", tuple(pico_common[index] for index in (1, 4, 10))),
        ("picoScan120", (pico_common[2],)),
    ):
        for profile in profiles:
            result.append(_sick_profile(
                config="SICK_picoScan100", asset_uri="/Isaac/Sensors/SICK/picoScan100/SICK_picoScan100.usd",
                model="picoScan100", product=product, profile=profile,
                function="Industrial 2D collision avoidance, measurement, and monitoring.",
            ))
    return tuple(result)


_DIRECT_PROFILES = (
    LidarProfile(
        "hesai-xt32-sd10", "HESAI XT32 SD10 · 32ch 10Hz", None, 32, 2000, 10.0,
        config="HESAI_XT32_SD10", asset_uri="/Isaac/Sensors/HESAI/XT32_SD10/HESAI_XT32_SD10.usd",
        manufacturer="HESAI", sensor_model="XT32 SD10", scan_type="rotary_3d",
        function="Mid-range 360-degree 3D ranging for robotics and autonomous platforms.",
        features="32-channel spinning LiDAR; up to two returns; manufacturer model asset.",
        profile_summary="32 channels · 10 Hz · 64,000 nominal rays/scan",
        minimum_range_m=0.05, maximum_range_m=120.0, nominal_ray_count=64_000,
    ),
    LidarProfile(
        "ouster-vls-128", "Velodyne VLS-128 (Ouster asset) · 128ch 10Hz", None, 128, 1876, 10.0,
        config="Ouster_VLS_128", asset_uri="/Isaac/Sensors/Ouster/VLS_128/Ouster_VLS_128.usd",
        manufacturer="Velodyne (Ouster)", sensor_model="VLS-128", scan_type="rotary_3d",
        function="Long-range, high-density 360-degree autonomous-vehicle perception.",
        features="128-channel spinning LiDAR with up to two returns.",
        profile_summary="128 channels · 10 Hz · high-density native rotary pattern",
        minimum_range_m=1.0, maximum_range_m=200.0,
    ),
    LidarProfile(
        "sick-nanoscan3", "SICK nanoScan3", None, 1, 1, 33.0,
        config="SICK_nanoScan3", asset_uri="/Isaac/Sensors/SICK/nanoScan3/SICK_nanoScan3.usd",
        manufacturer="SICK", sensor_model="nanoScan3", scan_type="rotary_2d",
        function="Compact safety laser scanning for mobile-platform protection and localization.",
        features="Manufacturer-certified 2D safety scanner with 0.17° angular resolution.",
        profile_summary="33 Hz · 0.17° angular resolution · native USD ray pattern",
        minimum_range_m=0.0, maximum_range_m=500.0, certified=True,
        range_summary="Read from the official nanoScan3 USD at runtime",
    ),
    LidarProfile(
        "sick-tim781", "SICK TiM781", None, 1, 1, 15.0,
        config="SICK_TIM781", asset_uri="/Isaac/Sensors/SICK/TIM781/SICK_TIM781.usd",
        manufacturer="SICK", sensor_model="TiM781", scan_type="rotary_2d",
        function="2D collision protection, object measurement, and area monitoring.",
        features="Manufacturer-certified industrial 2D LiDAR.",
        profile_summary="Native certified scan pattern",
        minimum_range_m=0.0, maximum_range_m=500.0, certified=True,
        range_summary="Read from the official TiM781 USD at runtime",
    ),
    LidarProfile(
        "slamtec-rplidar-s2e", "SLAMTEC RPLIDAR S2E · 10Hz", None, 1, 3200, 10.0,
        config="Slamtec_RPLIDAR_S2E", asset_uri="/Isaac/Sensors/Slamtec/RPLIDAR_S2E/Slamtec_RPLIDAR_S2E.usd",
        manufacturer="SLAMTEC", sensor_model="RPLIDAR S2E", scan_type="rotary_2d",
        function="Low-cost 360-degree 2D mapping and navigation.",
        features="Single-plane spinning scanner with one return.",
        profile_summary="3,200 nominal samples/scan · 10 Hz",
        minimum_range_m=0.05, maximum_range_m=30.0, nominal_ray_count=3200,
    ),
    LidarProfile(
        "zvision-ml30s", "ZVISION ML-30s+ · 50k 10Hz", None, 51200, 1, 10.0,
        config="ZVISION_ML30S", asset_uri="/Isaac/Sensors/ZVISION/ZVISION_ML30S.usda",
        manufacturer="ZVISION", sensor_model="ML-30s+", scan_type="solid_state",
        function="Short-range automotive-grade forward 3D perception without mechanical rotation.",
        features=(
            "Manufacturer-certified solid-state pattern; 905 nm; up to two returns. "
            "Isaac Sim 6.0.1 can produce sparse/empty output with this stock asset; use ML-Xs for the verified solid-state path."
        ),
        profile_summary="51,200 fixed-pattern rays/scan · 10 Hz",
        minimum_range_m=0.2, maximum_range_m=45.0, nominal_ray_count=51_200, certified=True,
    ),
    LidarProfile(
        "zvision-mlxs", "ZVISION ML-Xs · 108k 10Hz", None, 108000, 1, 10.0,
        config="ZVISION_MLXS", asset_uri="/Isaac/Sensors/ZVISION/ZVISION_MLXS.usda",
        manufacturer="ZVISION", sensor_model="ML-Xs", scan_type="solid_state",
        function="Long-range automotive-grade forward 3D perception without mechanical rotation.",
        features="Manufacturer-certified solid-state pattern; 1550 nm; up to two returns.",
        profile_summary="108,000 fixed-pattern rays/scan · 10 Hz",
        minimum_range_m=0.5, maximum_range_m=250.0, nominal_ray_count=108_000, certified=True,
    ),
)


LIDAR_PROFILES = (
    _DIRECT_PROFILES[0],
    *_ouster_profiles(),
    _DIRECT_PROFILES[1],
    *_sick_profiles(),
    _DIRECT_PROFILES[2],
    _DIRECT_PROFILES[3],
    _DIRECT_PROFILES[4],
    _DIRECT_PROFILES[5],
    _DIRECT_PROFILES[6],
)

if len(LIDAR_PROFILES) != OFFICIAL_NON_NVIDIA_PROFILE_COUNT:
    raise RuntimeError(f"official non-NVIDIA LiDAR catalog is incomplete: {len(LIDAR_PROFILES)} profiles")
if len({profile.profile_id for profile in LIDAR_PROFILES}) != len(LIDAR_PROFILES):
    raise RuntimeError("official non-NVIDIA LiDAR catalog contains duplicate profile ids")


# Isaac Sim 6.0.1 / RTX 3060 Ti catalog gate, 2026-09-07.
LIDAR_UNAVAILABLE_PROFILE_IDS = frozenset({
    "sick-lms5xx-lms5xxextendedrange-profile16-25hz-0p042deg",
    "sick-lms5xx-lms5xxprohr-profile16-25hz-0p042deg",
    "sick-lms5xx-lms5xxprosr-profile16-25hz-0p042deg",
    "zvision-ml30s",
})
LIDAR_UNAVAILABLE_REASON = (
    "N/A in the Isaac Sim 6.0.1 baseline: the official profile created, but "
    "produced only one return throughout the 180-update isolated "
    "closed-enclosure usability gate."
)


def _scanner_profile_is_selectable(profile: LidarProfile) -> bool:
    """Expose only the release-supported OS0 REV6 and REV7 profiles."""

    if profile.profile_id in LIDAR_UNAVAILABLE_PROFILE_IDS:
        return False
    return profile.config == "OS0" and profile.sensor_model in {"OS0 REV6", "OS0 REV7"}


# Preserve the complete official catalog for runtime lookup/capability probes,
# while limiting the Scanner product picker to the supported OS0 revisions.
SCANNER_LIDAR_PROFILES = tuple(
    profile for profile in LIDAR_PROFILES if _scanner_profile_is_selectable(profile)
)
SCANNER_EXCLUDED_PROFILE_IDS = frozenset(
    profile.profile_id for profile in LIDAR_PROFILES if not _scanner_profile_is_selectable(profile)
)

if len(SCANNER_LIDAR_PROFILES) != SCANNER_LIDAR_PROFILE_COUNT:
    raise RuntimeError(
        f"scanner LiDAR catalog contains {len(SCANNER_LIDAR_PROFILES)} profiles; "
        f"expected {SCANNER_LIDAR_PROFILE_COUNT} multi-line/3D profiles"
    )
if len(SCANNER_EXCLUDED_PROFILE_IDS) != SCANNER_EXCLUDED_PROFILE_COUNT:
    raise RuntimeError(
        "scanner LiDAR exclusion set contains "
        f"{len(SCANNER_EXCLUDED_PROFILE_IDS)} profiles; "
        f"expected {SCANNER_EXCLUDED_PROFILE_COUNT} excluded profiles"
    )
if DEFAULT_LIDAR_PROFILE_ID not in {profile.profile_id for profile in SCANNER_LIDAR_PROFILES}:
    raise RuntimeError("default LiDAR profile must remain selectable")

if not LIDAR_UNAVAILABLE_PROFILE_IDS <= {profile.profile_id for profile in LIDAR_PROFILES}:
    raise RuntimeError("LiDAR unavailable-profile baseline contains an unknown profile id")


def lidar_profile(profile_id: str) -> LidarProfile:
    for profile in LIDAR_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    raise KeyError(f"unknown LiDAR profile: {profile_id}")


def lidar_profile_is_available(profile_id: str) -> bool:
    """Return baseline application usability while rejecting unknown IDs."""

    lidar_profile(profile_id)
    return profile_id not in LIDAR_UNAVAILABLE_PROFILE_IDS


def lidar_profile_display_label(profile: LidarProfile) -> str:
    suffix = "" if lidar_profile_is_available(profile.profile_id) else " (N/A)"
    return f"{profile.label}{suffix}"


def lidar_profile_description_rows(profile: LidarProfile) -> tuple[tuple[str, str], ...]:
    rows = list(profile.description_rows)
    if not lidar_profile_is_available(profile.profile_id):
        name, features = rows[-1]
        rows[-1] = (name, f"{features} {LIDAR_UNAVAILABLE_REASON}")
    return tuple(rows)


@dataclass(frozen=True)
class LidarCapability:
    available: bool
    reason: str
    extension_version: str = LIDAR_EXTENSION_VERSION
    asset_uri: str = LIDAR_ASSET_URI


def probe_lidar_capability() -> LidarCapability:
    module = "isaacsim.sensors.experimental.rtx"
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError):
        spec = None
    if spec is None:
        return LidarCapability(False, f"missing Python API: {module}")
    return LidarCapability(True, "Lidar/LidarSensor/GenericModelOutput APIs are importable")


def _readonly_array(value: Any, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype).reshape(-1))
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class LidarNativeScan:
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    range_m: np.ndarray
    timestamp_ns: int
    emitted_ray_count: int
    intensity: np.ndarray | None = None
    material_id: np.ndarray | None = None
    time_offset_ns: np.ndarray | None = None

    def __post_init__(self) -> None:
        azimuth = _readonly_array(self.azimuth_deg, dtype=np.float32)
        elevation = _readonly_array(self.elevation_deg, dtype=np.float32)
        ranges = _readonly_array(self.range_m, dtype=np.float32)
        if not (azimuth.size == elevation.size == ranges.size):
            raise ValueError("LiDAR azimuth/elevation/range lengths must match")
        if self.timestamp_ns < 0 or self.emitted_ray_count <= 0:
            raise ValueError("LiDAR timestamp and emitted count are invalid")
        object.__setattr__(self, "azimuth_deg", azimuth)
        object.__setattr__(self, "elevation_deg", elevation)
        object.__setattr__(self, "range_m", ranges)
        for name, dtype in (("intensity", np.float32), ("material_id", np.uint32), ("time_offset_ns", np.int64)):
            value = getattr(self, name)
            if value is not None:
                array = _readonly_array(value, dtype=dtype)
                if array.size != ranges.size:
                    raise ValueError(f"LiDAR {name} length must match range length")
                object.__setattr__(self, name, array)

    @property
    def returned_count(self) -> int:
        return int(self.range_m.size)

    def points_sensor_m(self) -> np.ndarray:
        azimuth = np.deg2rad(self.azimuth_deg.astype(np.float64))
        elevation = np.deg2rad(self.elevation_deg.astype(np.float64))
        horizontal = self.range_m.astype(np.float64) * np.cos(elevation)
        return np.column_stack((horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), self.range_m * np.sin(elevation)))


class LidarBackend(Protocol):
    async def start(self) -> None: ...
    def read_scan(self) -> LidarNativeScan | None: ...
    def pose_world_m(self) -> Pose: ...
    def range_bounds_m(self) -> tuple[float, float]: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def quiesce_native_updates(self) -> None: ...
    def resume_native_updates(self) -> None: ...
    def close(self) -> None: ...


LidarBackendFactory = Callable[[LidarProfile, str], LidarBackend]


@dataclass(frozen=True)
class LidarRuntimeTelemetry:
    backend_creations: int = 0
    backend_closes: int = 0
    poll_attempts: int = 0
    no_data_scans: int = 0
    priming_discarded_scans: int = 0
    pose_discarded_scans: int = 0
    incomplete_scans: int = 0
    stale_scans: int = 0
    complete_scans: int = 0
    empty_complete_scans: int = 0
    published_frames: int = 0
    emitted_rays: int = 0
    returned_points: int = 0
    valid_points: int = 0
    last_timestamp_ns: int | None = None
    last_error: str | None = None


class RtxLidarRuntime:
    """Complete-scan orchestration over the common provider/dispatcher."""

    def __init__(self, *, stage_metadata: StageMetadata = StageMetadata(), dispatcher: ScanDispatcher | None = None,
                 backend_factory: LidarBackendFactory | None = None,
                 next_update: Callable[[], Awaitable[None]] | None = None,
                 generation_seed: int = 0,
                 diagnostics: DepthSensorDiagnostics = DepthSensorDiagnostics()) -> None:
        self._profiles = {profile.profile_id: profile for profile in LIDAR_PROFILES}
        self.provider = SensorProviderStateMachine(adapter_name="isaacsim-6.0.1-rtx-lidar-gmo",
            profiles=tuple(profile.descriptor for profile in LIDAR_PROFILES), dispatcher=dispatcher,
            stage_metadata=stage_metadata, generation_seed=generation_seed)
        self._backend_factory = backend_factory or IsaacRtxLidarBackend
        self._next_update = next_update or self._kit_next_update
        self._backend: LidarBackend | None = None
        self._active_profile: LidarProfile | None = None
        self._runtime_path: str | None = None
        self._last_timestamp_ns: int | None = None
        self._native_serial = 0
        self._latest_native_scan: LidarNativeScan | None = None
        self._no_returns = False
        self._discard_reasons: set[str] = set()
        self._telemetry = LidarRuntimeTelemetry()
        self._operation_lock = asyncio.Lock()
        self._diagnostics = diagnostics

    def configure_diagnostics(self, diagnostics: DepthSensorDiagnostics) -> None:
        """Apply common application-side scan diagnostics to later LiDAR scans."""

        self._diagnostics = diagnostics

    @property
    def dispatcher(self) -> ScanDispatcher:
        return self.provider.dispatcher

    def list_profiles(self) -> Sequence[SensorProfileDescriptor]:
        return self.provider.list_profiles()

    def latest_status(self) -> SensorStatus:
        return self.provider.latest_status()

    def latest_frame(self) -> ScanFrame | None:
        return self.provider.latest_frame()

    def latest_native_scan(self) -> LidarNativeScan | None:
        return self._latest_native_scan

    def has_no_returns(self) -> bool:
        """Whether the latest accepted complete scan contained zero returns."""

        return self._no_returns

    def telemetry(self) -> LidarRuntimeTelemetry:
        return self._telemetry

    def provider_telemetry(self) -> ProviderTelemetry:
        return self.provider.telemetry()

    async def start(self, profile_id: str, pose_world: Pose, runtime_path: str, *, maximum_warmup_updates: int = 600) -> SensorStatus:
        if maximum_warmup_updates <= 0:
            raise ValueError("maximum_warmup_updates must be positive")
        async with self._operation_lock:
            profile = self._profiles[profile_id]
            generation = self.provider.start(profile.descriptor, pose_world)
            self._runtime_path, self._active_profile = runtime_path, profile
            backend = self._backend_factory(profile, runtime_path)
            self._backend = backend
            self._no_returns = False
            self._telemetry = replace(self._telemetry, backend_creations=self._telemetry.backend_creations + 1, last_error=None)
            try:
                await backend.start()
                if not self.provider.resource_created(generation):
                    raise RuntimeError("LiDAR creation completed for a stale generation")
                self._schedule_accumulation_priming()
                ready, _frame = await self._warmup(generation, maximum_warmup_updates)
                if not ready:
                    raise RuntimeError(f"{profile_id} produced no valid complete scan in {maximum_warmup_updates} updates")
            except Exception as error:
                self._close_backend()
                self.provider.fail(generation, str(error))
                self._telemetry = replace(self._telemetry, last_error=str(error))
            return self.latest_status()

    def poll(self, *, timestamp_s: float | None = None, operation: SensorOperationToken | None = None) -> ScanFrame | None:
        backend, profile = self._backend, self._active_profile
        if backend is None or profile is None:
            return None
        clock = time.monotonic() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(clock) or clock < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        self._telemetry = replace(self._telemetry, poll_attempts=self._telemetry.poll_attempts + 1)
        scan = backend.read_scan()
        if scan is None:
            self._telemetry = replace(self._telemetry, no_data_scans=self._telemetry.no_data_scans + 1)
            return None
        if self._discard_reasons:
            reasons = self._discard_reasons
            self._discard_reasons = set()
            self._telemetry = replace(
                self._telemetry,
                priming_discarded_scans=(
                    self._telemetry.priming_discarded_scans
                    + ("priming" in reasons)
                ),
                pose_discarded_scans=(
                    self._telemetry.pose_discarded_scans
                    + ("pose" in reasons)
                ),
            )
            return None
        if self._last_timestamp_ns is not None and scan.timestamp_ns <= self._last_timestamp_ns:
            self._telemetry = replace(self._telemetry, stale_scans=self._telemetry.stale_scans + 1)
            return None
        self._last_timestamp_ns = scan.timestamp_ns
        self._latest_native_scan = scan
        if scan.returned_count == 0:
            status = self.provider.latest_status()
            accepted = self.provider.accept_empty_scan(
                generation=status.generation,
                operation=operation,
            )
            if accepted:
                self._no_returns = True
                self._telemetry = replace(
                    self._telemetry,
                    complete_scans=self._telemetry.complete_scans + 1,
                    empty_complete_scans=self._telemetry.empty_complete_scans + 1,
                    emitted_rays=self._telemetry.emitted_rays + scan.emitted_ray_count,
                    last_timestamp_ns=scan.timestamp_ns,
                )
            return None
        self._native_serial += 1
        status = self.provider.latest_status()
        range_bounds = getattr(backend, "range_bounds_m", None)
        minimum_range_m, maximum_range_m = (
            range_bounds() if range_bounds is not None
            else (profile.minimum_range_m, profile.maximum_range_m)
        )
        native = NativePointPayload(points_native=scan.points_sensor_m(), timestamp_s=clock,
            native_frame_id=f"lidar-{status.generation}-{scan.timestamp_ns}-{self._native_serial}",
            native_units="GenericModelOutput range in metres; angles converted from degrees",
            native_axes="OmniLidar SENSOR frame: +X forward, +Y left, +Z up", metres_per_native_unit=1.0,
            minimum_range_m=minimum_range_m, maximum_range_m=maximum_range_m,
            axial_error_sigma_m=self._diagnostics.effective_axial_error_sigma_m,
            axial_error_model=self._diagnostics.axial_error_model,
            axial_error_slope_m_per_m=self._diagnostics.effective_distance_error_parameters[0],
            axial_error_intercept_m=self._diagnostics.effective_distance_error_parameters[1],
            axial_error_seed=(status.generation << 32) | self._native_serial,
            details={"annotator": LIDAR_ANNOTATOR, "complete_scan": True,
                "config": profile.config, "variant": profile.variant_argument,
                "scan_type": profile.scan_type,
                "timestamp_ns": scan.timestamp_ns, "emitted_ray_count": scan.emitted_ray_count,
                "returned_point_count": scan.returned_count,
                "application_axial_error_enabled": self._diagnostics.axial_error_enabled,
                "application_axial_error_configured_sigma_m": self._diagnostics.axial_error_sigma_m,
                "application_axial_error_configured_model": self._diagnostics.axial_error_model,
                "application_axial_error_configured_slope_m_per_m": self._diagnostics.axial_error_slope_m_per_m,
                "application_axial_error_configured_intercept_m": self._diagnostics.axial_error_intercept_m,
                "native_aux_fields": tuple(name for name in ("intensity", "material_id", "time_offset_ns") if getattr(scan, name) is not None),
                "native_scan_retained": True})
        frame = self.provider.ingest_native(native, generation=status.generation, operation=operation,
            sensor_pose_world=backend.pose_world_m())
        if frame is not None:
            self._no_returns = False
        self._telemetry = replace(self._telemetry, complete_scans=self._telemetry.complete_scans + 1,
            published_frames=self._telemetry.published_frames + (frame is not None),
            emitted_rays=self._telemetry.emitted_rays + scan.emitted_ray_count,
            returned_points=self._telemetry.returned_points + scan.returned_count,
            valid_points=self._telemetry.valid_points + (0 if frame is None else frame.compacted_count),
            last_timestamp_ns=scan.timestamp_ns)
        return frame

    def disable(self) -> SensorStatus:
        self._require_backend().pause()
        self.provider.disable()
        return self.latest_status()

    def suspend_for_switch(self) -> None:
        """Quiesce native rendering while retaining this runtime for rollback."""

        backend = self._require_backend()
        if self.latest_status().lifecycle is SensorLifecycle.RUNNING:
            backend.pause()
        backend.quiesce_native_updates()

    def resume_after_failed_switch(self) -> None:
        """Reject reuse after native quiescence; the coordinator must recreate it."""

        raise RuntimeError("a switch-suspended RTX LiDAR backend must be recreated")

    def invalidate_pose(self) -> None:
        """Drop one complete scan after an external parent-pose transaction."""

        self._discard_reasons.add("pose")
        self._last_timestamp_ns = None
        self.provider.dispatcher.clear_generation(self.latest_status().generation)
        self._no_returns = False

    def invalidate_scene(self) -> None:
        """Discard an accumulated sweep that can include pre-edit geometry."""
        self._discard_reasons.add("scene")
        self._last_timestamp_ns = None

    async def enable(self, *, maximum_warmup_updates: int = 600) -> SensorStatus:
        async with self._operation_lock:
            self._require_backend().resume()
            self.provider.enable()
            self._last_timestamp_ns = None
            self._no_returns = False
            self._schedule_accumulation_priming()
            generation = self.latest_status().generation
            ready, _frame = await self._warmup(generation, maximum_warmup_updates)
            if not ready:
                message = "RTX LiDAR enable produced no complete scan"
                self.provider.fail(generation, message)
                self._telemetry = replace(self._telemetry, last_error=message)
            return self.latest_status()

    async def scan_once(self, *, maximum_warmup_updates: int = 600) -> ScanFrame | None:
        async with self._operation_lock:
            backend = self._require_backend()
            token = self.provider.scan_once()
            backend.resume()
            self._last_timestamp_ns = None
            self._no_returns = False
            self._schedule_accumulation_priming()
            try:
                _ready, frame = await self._warmup(
                    token.generation,
                    maximum_warmup_updates,
                    operation=token,
                )
                return frame
            finally:
                backend.pause()

    async def replace(self, profile_id: str, *, maximum_warmup_updates: int = 600) -> SensorStatus:
        async with self._operation_lock:
            profile = self._profiles[profile_id]
            was_disabled = self.latest_status().lifecycle is SensorLifecycle.DISABLED
            generation = self.provider.replace(profile.descriptor)
            self._close_backend()
            await self._next_update(); await self._next_update()
            if not self.provider.resource_stopped(generation):
                return self.latest_status()
            if self._runtime_path is None:
                raise RuntimeError("LiDAR runtime path is unavailable")
            backend = self._backend_factory(profile, self._runtime_path)
            self._backend, self._active_profile = backend, profile
            self._last_timestamp_ns = None
            self._no_returns = False
            self._telemetry = replace(self._telemetry, backend_creations=self._telemetry.backend_creations + 1, last_error=None)
            try:
                await backend.start(); self.provider.resource_created(generation)
                self._schedule_accumulation_priming()
                ready, _frame = await self._warmup(generation, maximum_warmup_updates)
                if not ready:
                    raise RuntimeError(f"{profile_id} produced no complete scan after recreate")
                if was_disabled:
                    backend.pause(); self.provider.disable()
            except Exception as error:
                self._close_backend(); self.provider.fail(generation, str(error))
                self._telemetry = replace(self._telemetry, last_error=str(error))
            return self.latest_status()

    def close(self) -> None:
        self._close_backend(); self.provider.close()
        self._active_profile = None; self._runtime_path = None; self._last_timestamp_ns = None
        self._discard_reasons.clear(); self._no_returns = False

    def _schedule_accumulation_priming(self) -> None:
        """Reject the first GMO publication after attaching an accumulated writer.

        An RTX LiDAR render product can be attached part-way through its scan
        period.  Its first accumulated publication therefore represents only
        the remainder of that period even though GMO exposes it as an ordinary
        publication.  Waiting for the following boundary prevents a startup or
        Frozen one-shot from preserving a visibly partial sweep.
        """

        self._discard_reasons.add("priming")

    async def _warmup(
        self,
        generation: int,
        maximum_updates: int,
        *,
        operation: SensorOperationToken | None = None,
    ) -> tuple[bool, ScanFrame | None]:
        for _ in range(maximum_updates):
            if generation != self.latest_status().generation:
                return False, None
            frame = self.poll(operation=operation)
            if frame is not None:
                return True, frame
            if self._no_returns:
                return True, None
            await self._next_update()
        return False, None

    def _close_backend(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            try: backend.close()
            finally: self._telemetry = replace(self._telemetry, backend_closes=self._telemetry.backend_closes + 1)

    def _require_backend(self) -> LidarBackend:
        if self._backend is None:
            raise RuntimeError("LiDAR backend is unavailable")
        return self._backend

    @staticmethod
    async def _kit_next_update() -> None:
        import omni.kit.app
        await omni.kit.app.get_app().next_update_async()


def _remove_prim_spec(layer: Any, path: Any) -> None:
    spec = layer.GetPrimAtPath(path)
    if spec is not None:
        del spec.nameParent.nameChildren[spec.name]


class IsaacRtxLidarBackend:
    """Kit-only official RTX LiDAR backend with anonymous-layer ownership."""

    def __init__(self, profile: LidarProfile, runtime_path: str) -> None:
        self._profile, self._runtime_path = profile, runtime_path
        self._stage = self._layer = self._previous_edit_target = None
        self._lidar = self._sensor = self._sensor_class = None
        self._writer = None
        self._writer_attached = False
        self._sensor_prim_path: str | None = None
        self._emitted_ray_count = profile.emitted_rays_per_scan
        self._range_bounds_m = (profile.minimum_range_m, profile.maximum_range_m)

    async def start(self) -> None:
        import omni.kit.app, omni.timeline, omni.usd
        from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor
        from pxr import Gf, Sdf, Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("a USD stage is required before starting RTX LiDAR")
        parent = Sdf.Path(self._runtime_path).GetParentPath()
        if not stage.GetPrimAtPath(parent):
            raise RuntimeError(f"LiDAR owner path is missing: {parent}")
        if stage.GetPrimAtPath(self._runtime_path):
            raise RuntimeError(f"LiDAR runtime path already exists: {self._runtime_path}")
        layer = Sdf.Layer.CreateAnonymous("FindSurfaceRtxLidarRuntime.usda")
        self._stage, self._layer, self._previous_edit_target = stage, layer, stage.GetEditTarget()
        stage.GetSessionLayer().subLayerPaths.append(layer.identifier)
        with Usd.EditContext(stage, layer):
            mount = UsdGeom.Xform.Define(stage, self._runtime_path)
            mount.GetPrim().SetCustomDataByKey("findSurfaceLidarMountOwned", True)
            x, y, z, w = LIDAR_TO_CAMERA_MOUNT_ORIENTATION_XYZW
            mount.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Quatd(w, Gf.Vec3d(x, y, z))
            )
            device_path = f"{self._runtime_path}/Device"
            self._lidar = Lidar.create(path=device_path, config=self._profile.config,
                variant=self._profile.variant_argument, accumulate_outputs=True, aux_output_level="FULL",
                attributes={"omni:sensor:Core:outputFrameOfReference": "SENSOR"},
                reset_xform_op_properties=False)
            self._sensor_prim_path = str(self._lidar.paths[0])
            sensor_prim = stage.GetPrimAtPath(self._sensor_prim_path)
            sensor_prim.SetCustomDataByKey("findSurfaceLidarRuntimeOwned", True)
            self._read_sensor_contract(sensor_prim)
        for _ in range(30): await omni.kit.app.get_app().next_update_async()
        self._sensor_class = LidarSensor
        self._create_sensor()
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing(): timeline.play()

    def read_scan(self) -> LidarNativeScan | None:
        if self._writer is None: return None
        scan, self._writer.latest = self._writer.latest, None
        return scan

    def pose_world_m(self) -> Pose:
        from pxr import Usd, UsdGeom
        if self._stage is None or self._sensor_prim_path is None: raise RuntimeError("LiDAR prim is unavailable")
        matrix = UsdGeom.Xformable(self._stage.GetPrimAtPath(self._sensor_prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation, quaternion = matrix.ExtractTranslation(), matrix.ExtractRotationQuat(); imaginary = quaternion.GetImaginary()
        metadata = StageMetadata(str(UsdGeom.GetStageUpAxis(self._stage)), float(UsdGeom.GetStageMetersPerUnit(self._stage)))
        return StageSpace(metadata).to_world_pose(StagePose(tuple(float(translation[i]) for i in range(3)),
            tuple(float(imaginary[i]) for i in range(3)) + (float(quaternion.GetReal()),)))

    def range_bounds_m(self) -> tuple[float, float]:
        return self._range_bounds_m

    def pause(self) -> None:
        if self._writer is not None:
            self._writer.latest = None
            if self._writer_attached:
                self._writer.detach()
                self._writer_attached = False

    def resume(self) -> None:
        if self._lidar is None: raise RuntimeError("cannot resume a closed LiDAR")
        if self._sensor is None:
            self._create_sensor()
        elif self._writer is not None and not self._writer_attached:
            self._writer.latest = None
            self._writer.attach([str(self._sensor.render_product.GetPath())])
            self._writer_attached = True

    def quiesce_native_updates(self) -> None:
        self._set_native_updates_enabled(False)

    def resume_native_updates(self) -> None:
        self._set_native_updates_enabled(True)

    def _set_native_updates_enabled(self, enabled: bool) -> None:
        sensor = self._sensor
        hydra_texture = None if sensor is None else getattr(sensor, "_hydra_texture", None)
        if hydra_texture is not None:
            native_texture = getattr(hydra_texture, "hydra_texture", hydra_texture)
            native_texture.set_updates_enabled(enabled)

    def close(self) -> None:
        from pxr import Sdf
        self.pause()
        sensor, self._sensor = self._sensor, None
        if sensor is not None: sensor._invalidate_sensor()
        self._writer = None
        self._lidar = None; self._sensor_prim_path = None
        stage, self._stage = self._stage, None; layer, self._layer = self._layer, None
        if stage is None or layer is None: return
        if self._previous_edit_target is not None: stage.SetEditTarget(self._previous_edit_target)
        self._previous_edit_target = None
        paths = list(stage.GetSessionLayer().subLayerPaths)
        if layer.identifier in paths: paths.remove(layer.identifier); stage.GetSessionLayer().subLayerPaths = paths
        path = Sdf.Path(self._runtime_path); _remove_prim_spec(stage.GetSessionLayer(), path); _remove_prim_spec(stage.GetRootLayer(), path)

    def _create_sensor(self) -> None:
        if self._sensor_class is None or self._lidar is None: raise RuntimeError("LiDAR API is unavailable")
        import omni.replicator.core as rep
        from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data
        import isaacsim.sensors.experimental.rtx.generic_model_output as generic_model_output
        from omni.replicator.core import Writer

        emitted_ray_count = self._emitted_ray_count

        class _LatestGmoWriter(Writer):
            def __init__(self) -> None:
                self.data_structure = "renderProduct"
                self.annotators = [rep.annotators.get("GenericModelOutput")]
                self.latest = None

            def write(self, data: Any) -> None:
                for render_product in data.get("renderProducts", {}).values():
                    raw = render_product.get("GenericModelOutput")
                    if isinstance(raw, dict): raw = raw.get("data")
                    if raw is None: continue
                    gmo = parse_generic_model_output_data(raw)
                    if gmo.magicNumber != generic_model_output.getMagicNumberGMO():
                        continue
                    if gmo.numElements == 0:
                        self.latest = LidarNativeScan([], [], [], 0, emitted_ray_count)
                        continue
                    self.latest = LidarNativeScan(gmo.x.copy(), gmo.y.copy(), gmo.z.copy(), int(gmo.timestampNs),
                        max(emitted_ray_count, int(gmo.numElements)), intensity=gmo.scalar.copy(),
                        material_id=gmo.matId.copy(), time_offset_ns=gmo.timeOffsetNs.copy())

        self._sensor = self._sensor_class(self._lidar)
        self._writer = _LatestGmoWriter()
        self._writer.attach([str(self._sensor.render_product.GetPath())])
        self._writer_attached = True

    def _read_sensor_contract(self, prim: Any) -> None:
        def value(name: str, fallback: Any) -> Any:
            attribute = prim.GetAttribute(f"omni:sensor:Core:{name}")
            if not attribute:
                return fallback
            result = attribute.Get()
            return fallback if result is None else result

        near_m = float(value("nearRangeM", self._profile.minimum_range_m))
        far_m = float(value("farRangeM", self._profile.maximum_range_m))
        if math.isfinite(near_m) and math.isfinite(far_m) and 0.0 <= near_m < far_m:
            self._range_bounds_m = (near_m, far_m)

        # Isaac Sim 6.0 requires the sensor tick rate to equal the selected
        # asset/variant scanRateBaseHz for complete-scan accumulation.  Read it
        # from the composed prim instead of trusting catalog display metadata.
        scan_rate = float(value("scanRateBaseHz", self._profile.scan_rate_hz))
        tick_rate = prim.GetAttribute("omni:sensor:tickRate")
        if tick_rate and math.isfinite(scan_rate) and scan_rate > 0.0:
            tick_rate.Set(scan_rate)

        rays_per_line = tuple(int(item) for item in value("numRaysPerLine", ()) or ())
        if rays_per_line and all(item > 0 for item in rays_per_line):
            self._emitted_ray_count = sum(rays_per_line)
            return
        emitters = int(value("numberOfEmitters", 0))
        firing_rate = float(value("patternFiringRateHz", 0.0))
        if emitters > 0 and firing_rate > 0.0 and scan_rate > 0.0:
            self._emitted_ray_count = max(1, int(round(emitters * firing_rate / scan_rate)))


__all__ = ["DEFAULT_LIDAR_PROFILE_ID", "LIDAR_ANNOTATOR", "LIDAR_ASSET_URI", "LIDAR_PROFILES",
    "LIDAR_UNAVAILABLE_PROFILE_IDS", "LIDAR_UNAVAILABLE_REASON",
    "LIDAR_TO_CAMERA_MOUNT_ORIENTATION_XYZW",
    "OFFICIAL_NON_NVIDIA_CONFIG_COUNT", "OFFICIAL_NON_NVIDIA_PROFILE_COUNT",
    "SCANNER_EXCLUDED_PROFILE_COUNT", "SCANNER_EXCLUDED_PROFILE_IDS",
    "SCANNER_LIDAR_PROFILE_COUNT", "SCANNER_LIDAR_PROFILES",
    "IsaacRtxLidarBackend", "LidarCapability", "LidarNativeScan", "LidarProfile", "LidarRuntimeTelemetry",
    "RtxLidarRuntime", "lidar_profile", "lidar_profile_description_rows",
    "lidar_profile_display_label", "lidar_profile_is_available", "probe_lidar_capability"]
