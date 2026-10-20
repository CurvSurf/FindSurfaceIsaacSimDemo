"""Immutable, toolkit-neutral projection of one captured geometry card."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .geometry_comparison import ComparisonRow
    from .result_overlay import HistogramOverlayPresentation


@dataclass(frozen=True)
class GeometryCardPresentation:
    capture_id: str
    geometry_type: str
    mesh_visible: bool
    inliers_visible: bool
    selected: bool
    rows: tuple[tuple[str, str], ...]
    heatmap_basis: str
    histograms: tuple[tuple[str, HistogramOverlayPresentation], ...]
    comparison_rows: tuple[ComparisonRow, ...] = ()
    mesh_appearance: tuple = ()

    @property
    def structure_signature(self) -> tuple:
        return (self.capture_id, self.geometry_type, self.rows,
                self.histograms, self.comparison_rows)


def normalize_geometry_card_values(values: dict[str, object]) -> dict[str, object]:
    """Adapt legacy 8/9/10-field fixtures once, at the UI input boundary."""
    entries = values.get("geometry.entries", ())
    if isinstance(entries, tuple) and all(
        isinstance(card, GeometryCardPresentation) for card in entries
    ):
        return values
    cards = []
    for entry in entries:
        if isinstance(entry, GeometryCardPresentation):
            cards.append(entry)
            continue
        if not 8 <= len(entry) <= 10:
            raise ValueError("Legacy geometry cards must contain 8, 9 or 10 fields")
        capture_id, geometry_type, mesh_visible, inliers_visible, selected, rows, basis, histograms = entry[:8]
        cards.append(GeometryCardPresentation(
            capture_id=str(capture_id), geometry_type=geometry_type,
            mesh_visible=mesh_visible, inliers_visible=inliers_visible,
            selected=selected, rows=rows, heatmap_basis=basis, histograms=histograms,
            comparison_rows=entry[8] if len(entry) > 8 else (),
            mesh_appearance=entry[9] if len(entry) > 9 else (),
        ))
    return {**values, "geometry.entries": tuple(cards)}
