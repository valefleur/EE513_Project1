# camera_specs.py

from dataclasses import dataclass

@dataclass(frozen=True)
class CameraSpec:
    name: str
    focal_mm: float
    f_number: float
    pitch_um: float
    saturation_dn: int          # measured in HW4 Problem 1, not the declared white level
    black_level_dn: float | None = None   # None: use the black level declared in each file
    wavelength_um: float = 0.550

    def focal_px(self, bin_factor=1):
        """Focal length in (binned) pixels."""
        return self.focal_mm * 1000 / (self.pitch_um * bin_factor)

IPHONE_16_PRO_MAX = CameraSpec(
    name="iPhone 16 Pro Max main",
    focal_mm=6.765,
    f_number=1.78,
    pitch_um=2.434,
    saturation_dn=4095,
    # Declared BlackLevel in every DNG so far. HW3 dark frames at the shortest exposure
    # average 527.40 to 527.50 DN per channel (README), slightly below this.
    black_level_dn=528,
)