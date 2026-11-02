# camera_specs.py

from dataclasses import dataclass

@dataclass(frozen=True)
class CameraSpec:
    name: str
    focal_mm: float
    f_number: float
    pitch_um: float
    saturation_dn: int          # measured in HW4 Problem 1, not the declared white level
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
)