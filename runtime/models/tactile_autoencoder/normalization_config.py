from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ForceNormalizationV41Config:
    """Fixed, zero-preserving physical-force transform."""

    deadband_low_n: float = 1.0e-4
    deadband_high_n: float = 1.0e-3
    knee_n: tuple[float, float, float] = (0.01, 0.001, 0.001)
    upper_n: tuple[float, float, float] = (50.0, 5.0, 6.1)

    def __post_init__(self) -> None:
        if not 0 <= self.deadband_low_n < self.deadband_high_n:
            raise ValueError("deadband bounds must satisfy 0 <= low < high")
        if len(self.knee_n) != 3 or len(self.upper_n) != 3:
            raise ValueError("normalizer requires exactly three channel parameters")
        if any(value <= 0 for value in (*self.knee_n, *self.upper_n)):
            raise ValueError("normalizer knee and upper values must be positive")
