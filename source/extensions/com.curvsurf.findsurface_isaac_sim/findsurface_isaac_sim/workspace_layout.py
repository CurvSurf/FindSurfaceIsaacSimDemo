"""Small, versioned layout preference for product-owned property windows.

Kit's global workspace dump can omit dock nodes for our custom ToolBar. Only
capture values that the product can restore through its own docking sequence.
"""

from __future__ import annotations

import json
import math


PERSISTENT_WORKSPACE_PATH = "/persistent/exts/com.curvsurf.findsurface_isaac_sim/workspaceLayout"
WORKSPACE_LAYOUT_VERSION = 2
PROPERTY_TITLES = (
    "FindSurface", "Geometries", "Objects", "Camera", "Scanner", "Graphics", "Settings",
)
DEFAULT_PROPERTY_WIDTH = 490
MIN_FLOATING_WIDTH = 320
MIN_FLOATING_HEIGHT = 240
SIDE_PANEL_DOCK_RATIO = 0.29034810554922014


def default_property_dock_ratio(workspace_width: float) -> float:
    """Keep the default property panel readable without taking most of the view."""
    if workspace_width <= 0:
        return SIDE_PANEL_DOCK_RATIO
    return min(0.50, max(SIDE_PANEL_DOCK_RATIO, DEFAULT_PROPERTY_WIDTH / workspace_width))


def encode_workspace_layout(layout: dict) -> str:
    if decode_workspace_layout(json.dumps({"version": WORKSPACE_LAYOUT_VERSION, "layout": layout})) != layout:
        raise ValueError("invalid product window layout")
    return json.dumps({"version": WORKSPACE_LAYOUT_VERSION, "layout": layout}, separators=(",", ":"))


def fit_floating_bounds(bounds, content_size) -> list[float]:
    """Fit the complete floating window in logical client coordinates, without moving live UI."""
    x, y, width, height = map(float, bounds)
    available_width, available_height = map(float, content_size)
    if not all(math.isfinite(value) for value in (x, y, width, height, available_width, available_height)):
        raise ValueError("window bounds and content size must be finite")
    if min(width, height, available_width, available_height) <= 0:
        raise ValueError("window and content sizes must be positive")
    width = min(available_width, max(MIN_FLOATING_WIDTH, width))
    height = min(available_height, max(MIN_FLOATING_HEIGHT, height))
    return [min(max(0.0, x), available_width - width),
            min(max(0.0, y), available_height - height), width, height]


def fit_workspace_layout(layout: dict, content_size) -> dict:
    """Return a bounded persistence snapshot; never mutate the user's current layout."""
    result = dict(layout)
    result["floating"] = {title: fit_floating_bounds(bounds, content_size)
                          for title, bounds in layout["floating"].items()}
    # Reject unavailable/minimized content extents even when all windows are docked.
    fit_floating_bounds([0, 0, 1, 1], content_size)
    return result


def decode_workspace_layout(raw: str) -> dict | None:
    if not raw or len(raw) > 100_000:
        return None
    try:
        payload = json.loads(raw)
        if payload["version"] != WORKSPACE_LAYOUT_VERSION:
            return None
        layout = payload["layout"]
        ratio = layout["side_ratio"]
        tabs = layout["tab_order"]
        active = layout["active_tab"]
        floating = layout["floating"]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0.20 <= ratio <= 0.50:
            return None
        if not isinstance(tabs, list) or set(tabs) != set(PROPERTY_TITLES) or len(tabs) != len(PROPERTY_TITLES):
            return None
        if active is not None and active not in PROPERTY_TITLES:
            return None
        if not isinstance(floating, dict) or not set(floating) <= set(PROPERTY_TITLES):
            return None
        for bounds in floating.values():
            if not isinstance(bounds, list) or len(bounds) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds
            ) or bounds[2] <= 0 or bounds[3] <= 0:
                return None
        return layout
    except (KeyError, TypeError, ValueError):
        return None


__all__ = [
    "PERSISTENT_WORKSPACE_PATH", "PROPERTY_TITLES",
    "DEFAULT_PROPERTY_WIDTH", "SIDE_PANEL_DOCK_RATIO", "default_property_dock_ratio",
    "decode_workspace_layout", "encode_workspace_layout",
    "fit_floating_bounds", "fit_workspace_layout", "MIN_FLOATING_WIDTH", "MIN_FLOATING_HEIGHT",
]
