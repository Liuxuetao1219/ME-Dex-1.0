from dataclasses import dataclass


@dataclass(frozen=True)
class ForceNormalizationConfig:
    """Smooth deadband and signed-asinh force scaling."""

    deadband_low_n: float = 1.0e-4
    deadband_high_n: float = 1.0e-3
    knee_n: tuple[float, float, float] = (0.01, 0.001, 0.001)
    upper_n: tuple[float, float, float] = (50.0, 5.0, 6.1)
