"""Runtime capability reporting with no import-time dependency on Kit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Callable, Iterable


@dataclass(frozen=True)
class CapabilitySpec:
    key: str
    module: str
    purpose: str
    required_for_product: bool
    activation: str = "deferred"
    attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityStatus:
    key: str
    module: str
    purpose: str
    required_for_product: bool
    activation: str
    available: bool
    version: str | None
    missing_attributes: tuple[str, ...]
    error: str | None


@dataclass(frozen=True)
class CapabilityReport:
    schema_version: int
    runtime_baseline: str
    minimal_app_ready: bool
    product_ready: bool
    capabilities: tuple[CapabilityStatus, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CAPABILITY_SPECS = (
    CapabilitySpec(
        "opengl_widget",
        "omni.kit.widget.opengl",
        "scanner and scene OpenGL renderer",
        True,
        attributes=("OpenGLWidget",),
    ),
    CapabilitySpec("usd_stage", "omni.usd", "USD stage access", True),
    CapabilitySpec("timeline", "omni.timeline", "simulation timeline", True),
    CapabilitySpec(
        "depth_sensor",
        "isaacsim.sensors.experimental.rtx",
        "single-view depth camera",
        True,
        attributes=("SingleViewDepthCameraSensor",),
    ),
    CapabilitySpec(
        "findsurface_native",
        "findsurface.bootstrap",
        "FindSurface native detection SDK",
        True,
    ),
    CapabilitySpec(
        "rtx_lidar", "isaacsim.sensors.experimental.rtx", "experimental RTX LiDAR acquisition",
        False, attributes=("Lidar", "LidarSensor", "parse_generic_model_output_data"),
    ),
)


def _resolve_attribute(module: object, dotted_name: str) -> object:
    value = module
    for part in dotted_name.split("."):
        value = getattr(value, part)
    return value


def collect_capabilities(
    importer: Callable[[str], object] = import_module,
    specs: Iterable[CapabilitySpec] = CAPABILITY_SPECS,
    version_resolver: Callable[[str, object], str | None] | None = None,
) -> CapabilityReport:
    """Collect every capability independently; a missing item never aborts the probe."""

    statuses: list[CapabilityStatus] = []
    for spec in specs:
        available = False
        version = None
        missing_attributes: list[str] = []
        error = None
        try:
            module = importer(spec.module)
            for attribute in spec.attributes:
                try:
                    _resolve_attribute(module, attribute)
                except AttributeError:
                    missing_attributes.append(attribute)
            available = not missing_attributes
            version = getattr(module, "__version__", None)
            if version_resolver is not None:
                version = version_resolver(spec.module, module) or version
        except Exception as exception:  # diagnostics must survive partial runtimes
            error = f"{type(exception).__name__}: {exception}"
        statuses.append(
            CapabilityStatus(
                key=spec.key,
                module=spec.module,
                purpose=spec.purpose,
                required_for_product=spec.required_for_product,
                activation=spec.activation,
                available=available,
                version=version,
                missing_attributes=tuple(missing_attributes),
                error=error,
            )
        )

    product_ready = all(
        status.available for status in statuses if status.required_for_product
    )
    return CapabilityReport(
        schema_version=1,
        runtime_baseline="Isaac Sim 6.0.1",
        minimal_app_ready=True,
        product_ready=product_ready,
        capabilities=tuple(statuses),
    )
