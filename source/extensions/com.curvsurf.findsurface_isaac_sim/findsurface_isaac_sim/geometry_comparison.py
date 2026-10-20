"""Small immutable capture table; never consult the current Objects catalog."""
from dataclasses import dataclass
import math

from .domain import GeometryMode
from .object_orientation import orientation_axes


@dataclass(frozen=True)
class ComparisonRow:
    label: str
    detected: str
    original: str
    difference: str
    height: int = 26


def axis_angle_degrees(a, b, *, directed=False):
    denominator = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
    if denominator <= 0:
        raise ValueError("axis must be nonzero")
    cosine = sum(x*y for x, y in zip(a, b)) / denominator
    cosine = cosine if directed else abs(cosine)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _number(value, signed=False):
    # ASCII hyphen-minus, not U+2212. Avoid displaying rounded negative zero.
    value = float(value)
    if round(value, 4) == 0:
        value = 0.0
    return format(value, '+.4f' if signed else '.4f')


def _vector(value):
    return '\n'.join(f'{axis} {_number(v)}' for axis, v in zip('XYZ', value))


def _values(item):
    values = item.parameter_dict.copy()
    axis = orientation_axes(item.pose_world_m.orientation_xyzw)[2]
    # Cone's directed axis points toward the smaller-radius end. This is a
    # display-only canonicalization, never a modification of SDK input/output.
    if item.geometry_type is GeometryMode.CONE and values['top_radius_m'] > values['bottom_radius_m']:
        axis = tuple(-v for v in axis)
        values['top_radius_m'], values['bottom_radius_m'] = values['bottom_radius_m'], values['top_radius_m']
    return values, axis


def comparison_rows(detected, original):
    if original is not None and original.geometry_type is not detected.geometry_type:
        original = None
    detected_position = detected.pose_world_m.position_m
    original_position = None if original is None else original.pose_world_m.position_m
    position_difference = 'N/A'
    if original_position is not None:
        position_distance_m = math.dist(detected_position, original_position)
        position_difference = _number(position_distance_m) + ' m'
    rows = [ComparisonRow(
        label='Position (m)',
        detected=_vector(detected_position),
        original='N/A' if original_position is None else _vector(original_position),
        difference=position_difference,
        height=60,
    )]
    detected_parameters, detected_axis = _values(detected)
    original_parameters, original_axis = ({}, None) if original is None else _values(original)
    if detected.geometry_type is not GeometryMode.SPHERE:
        direction_difference = 'N/A'
        if original_axis is not None:
            axis_difference_deg = axis_angle_degrees(
                detected_axis, original_axis,
                directed=detected.geometry_type is GeometryMode.CONE,
            )
            direction_difference = _number(axis_difference_deg) + ' deg'
        rows.append(ComparisonRow(
            label='Normal' if detected.geometry_type is GeometryMode.PLANE else 'Axis',
            detected=_vector(detected_axis),
            original='N/A' if original_axis is None else _vector(original_axis),
            difference=direction_difference,
            height=60,
        ))
    labels = {'width_m': 'Width', 'height_m': 'Height', 'radius_m': 'Radius',
              'length_m': 'Length', 'top_radius_m': 'Top radius', 'bottom_radius_m': 'Bottom radius',
              'major_radius_m': 'Major radius', 'minor_radius_m': 'Tube radius', 'sweep_angle_deg': 'Sweep'}
    for key, label in labels.items():
        if key not in detected_parameters:
            continue
        unit = 'deg' if key.endswith('_deg') else 'm'
        detected_value = detected_parameters[key]
        original_value = original_parameters.get(key)
        parameter_difference = 'N/A'
        if original_value is not None:
            detected_minus_original = detected_value - original_value
            parameter_difference = _number(detected_minus_original, signed=True)
        rows.append(ComparisonRow(
            label=f'{label} ({unit})',
            detected=_number(detected_value),
            original='N/A' if original_value is None else _number(original_value),
            difference=parameter_difference,
        ))
    return tuple(rows)
