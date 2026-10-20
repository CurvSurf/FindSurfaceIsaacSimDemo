"""Pure mapping for the FindSurface parameter controls' eleven slider positions."""


def slider_index_for_value(value: float, minimum: float, maximum: float) -> int:
    if maximum <= minimum:
        return 0
    fraction_of_range = (value - minimum) / (maximum - minimum)
    rounded_index = int(round(fraction_of_range * 10))
    return max(0, min(10, rounded_index))
