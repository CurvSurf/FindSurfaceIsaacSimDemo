"""Debug-only XYZ export of an immutable point-cloud snapshot."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


def repository_root_from_repo_bat(start: str | Path | None = None) -> Path:
    """Find the exact directory containing this checkout's ``repo.bat``."""

    location = Path(__file__).resolve() if start is None else Path(start).resolve()
    if location.is_file():
        location = location.parent
    for candidate in (location, *location.parents):
        if (candidate / "repo.bat").is_file():
            return candidate
    raise FileNotFoundError(f"repo.bat was not found above {location}")


def xyz_export_filename(point_count: int, *, now: datetime | None = None) -> str:
    if point_count < 0:
        raise ValueError("point_count cannot be negative")
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S-%f")
    return f"pointcloud-{timestamp}-{point_count}pts.xyz"


def export_points_xyz(
    points_world_m: np.ndarray,
    *,
    directory: str | Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Write one world-metre ``x y z`` point per line without modifying points."""

    points = np.asarray(points_world_m)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("XYZ export requires an Nx3 point array")
    if not np.isfinite(points).all():
        raise ValueError("XYZ export requires finite point coordinates")
    destination_directory = (
        repository_root_from_repo_bat() if directory is None else Path(directory).resolve()
    )
    if not (destination_directory / "repo.bat").is_file():
        raise FileNotFoundError(f"export directory does not contain repo.bat: {destination_directory}")
    destination = destination_directory / xyz_export_filename(len(points), now=now)
    temporary = destination.with_suffix(".xyz.tmp")
    try:
        np.savetxt(temporary, points, fmt="%.9g %.9g %.9g", delimiter=" ", newline="\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


__all__ = ["export_points_xyz", "repository_root_from_repo_bat", "xyz_export_filename"]
