from enum import StrEnum, auto

class DownsampleMethod(StrEnum):
    # The numeric value is spread uniformly over the timeframe,
    # ensuring the total amount remains consistent.
    UNIFORM = auto()

    # The numeric value remains the same as the original,
    # without preserving the total amount across the timeframe.
    REPLICA = auto()
