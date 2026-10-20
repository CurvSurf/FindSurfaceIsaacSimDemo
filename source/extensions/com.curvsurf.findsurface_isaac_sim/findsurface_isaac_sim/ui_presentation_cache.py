"""Main-thread, bounded caches for immutable UI dependencies and projections."""

from dataclasses import dataclass
from typing import Callable


INTERACTION_PROPERTY_FIELDS = (
    "detection_average_window",
    "point_rendering_size", "inlier_rendering_size", "outlier_color_rgba",
    "ruler_main_opacity", "ruler_pip_opacity",
    "inlier_negative_rgba", "inlier_zero_rgba", "inlier_positive_rgba",
    "detected_preview_subdivision", "detected_preview_face_rgba", "detected_preview_wire_rgba",
    "detected_mesh_appearances", "captured_mesh_appearances", "probe_radius_px",
    "pick_nearest_on_probe_miss", "sticky_auto_preview", "sticky_preview_timeout_ms",
    "minimum_geometry_length_m", "surface_side_scale_mode", "surface_side_fixed_scale_m",
    "surface_side_sigma_multiplier",
)


def interaction_property_dependencies(settings):
    return tuple(getattr(settings, field) for field in INTERACTION_PROPERTY_FIELDS)


@dataclass(frozen=True, eq=False)
class IdentityDependency:
    value: object

    def __eq__(self, other):
        return isinstance(other, IdentityDependency) and self.value is other.value


def identity_dependency(value):
    return IdentityDependency(value)


class UiPresentationCache:
    def __init__(self):
        self._entries = {}
        self._sections = ()
        self._values = {}
        self.build_counts = {}

    def resolve(self, slot: str, dependencies: tuple, build: Callable):
        previous = self._entries.get(slot)
        if previous is not None and previous[0] == dependencies:
            return previous[1]
        value = build()
        self._entries[slot] = (dependencies, value)
        self.build_counts[slot] = self.build_counts.get(slot, 0) + 1
        return value

    def retain(self, prefix: str, slots):
        keep = set(slots)
        for slot in tuple(self._entries):
            if slot.startswith(prefix) and slot not in keep:
                del self._entries[slot]
                self.build_counts.pop(slot, None)

    def merge(self, sections):
        sections = tuple(sections)
        if len(sections) == len(self._sections) and all(
            current is previous for current, previous in zip(sections, self._sections)
        ):
            return self._values
        self._values = {key: value for section in sections for key, value in section.items()}
        self._sections = sections
        return self._values

    def clear(self):
        self._entries.clear()
        self._sections = ()
        self._values = {}
        self.build_counts.clear()
