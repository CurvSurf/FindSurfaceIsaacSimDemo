"""Host-safe RTX non-visual target-material catalog and recommendations."""

from __future__ import annotations

from dataclasses import dataclass


SENSOR_MATERIAL_BASES = (
    "none", "aluminum", "steel", "oxidized_steel", "iron", "oxidized_iron",
    "silver", "brass", "bronze", "oxidized_Bronze_Patina", "tin", "plastic",
    "fiberglass", "carbon_fiber", "vinyl", "plexiglass", "pvc", "nylon",
    "polyester", "clear_glass", "frosted_glass", "one_way_mirror", "mirror",
    "ceramic_glass", "asphalt", "concrete", "leaf_grass", "dead_leaf_grass",
    "rubber", "wood", "bark", "cardboard", "paper", "fabric", "skin",
    "fur_hair", "leather", "marble", "brick", "stone", "gravel", "dirt",
    "mud", "water", "salt_water", "snow", "ice", "calibration_lambertion",
)
SENSOR_MATERIAL_COATINGS = ("none", "paint", "clearcoat", "paint_clearcoat")
SENSOR_MATERIAL_ATTRIBUTES = (
    "none", "emissive", "retroreflective", "single_sided", "visually_transparent",
)


@dataclass(frozen=True)
class SensorMaterialSetting:
    """One material ID consumed by Isaac Sim RTX non-visual sensors."""

    base: str = "concrete"
    coating: str = "paint"
    attribute: str = "emissive"

    def __post_init__(self) -> None:
        if self.base not in SENSOR_MATERIAL_BASES:
            raise ValueError(f"unsupported sensor material base: {self.base}")
        if self.coating not in SENSOR_MATERIAL_COATINGS:
            raise ValueError(f"unsupported sensor material coating: {self.coating}")
        if self.attribute not in SENSOR_MATERIAL_ATTRIBUTES:
            raise ValueError(f"unsupported sensor material attribute: {self.attribute}")

    @property
    def summary(self) -> str:
        return f"{self.base} / {self.coating} / {self.attribute}"


DEPTH_RECOMMENDED_MATERIAL = SensorMaterialSetting("concrete", "paint", "emissive")


def recommended_lidar_material(manufacturer: str) -> SensorMaterialSetting:
    manufacturer = str(manufacturer).lower()
    if "sick" in manufacturer or "slamtec" in manufacturer:
        return SensorMaterialSetting("steel", "clearcoat", "emissive")
    if "zvision" in manufacturer:
        return SensorMaterialSetting("aluminum", "paint", "emissive")
    return SensorMaterialSetting("concrete", "paint", "emissive")


__all__ = [
    "DEPTH_RECOMMENDED_MATERIAL", "SENSOR_MATERIAL_ATTRIBUTES", "SENSOR_MATERIAL_BASES",
    "SENSOR_MATERIAL_COATINGS", "SensorMaterialSetting", "recommended_lidar_material",
]
