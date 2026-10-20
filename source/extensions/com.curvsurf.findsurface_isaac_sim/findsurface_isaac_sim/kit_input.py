"""Small, host-testable normalization helpers for Kit input enums."""

from __future__ import annotations


def kit_enum_name(value) -> str:
    """Return the stable member name across Kit Python enum wrappers."""

    return str(getattr(value, "name", value)).split(".")[-1].upper()


def kit_key_name(value) -> str:
    """Normalize both ``W`` and wrapper spellings such as ``KEY_W``."""

    name = kit_enum_name(value)
    if name.startswith("KEY_") and (len(name) == 5 or name[4:] in ("UP", "DOWN", "LEFT", "RIGHT")):
        return name[4:]
    return name


def kit_modifier_has_alt(value, input_api=None) -> bool:
    """Recognize Alt/Option in an omni.ui pointer modifier bit mask."""

    try:
        mask = int(value or 0)
    except (TypeError, ValueError):
        return False
    flags = [4]  # Kit's stable common ALT bit and demo-1 fallback.
    if input_api is not None:
        flags.extend(
            int(getattr(input_api, name, 0) or 0)
            for name in (
                "KEYBOARD_MODIFIER_FLAG_ALT",
                "KEYBOARD_MODIFIER_FLAG_LEFT_ALT",
                "KEYBOARD_MODIFIER_FLAG_RIGHT_ALT",
            )
        )
    return any(flag and mask & flag for flag in flags)


__all__ = ["kit_enum_name", "kit_key_name", "kit_modifier_has_alt"]
