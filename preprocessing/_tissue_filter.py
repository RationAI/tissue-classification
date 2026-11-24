"""Tissue-mask filter shared by ``tissue_masks.py`` and the standalone inference."""

from rationai.masks.vips_filters import (
    VipsClosing,
    VipsCompose,
    VipsGrayScaleFilter,
    VipsOpening,
    VipsOtsu,
)


def tissue_filter(disk_factor: int) -> VipsCompose:
    """Grayscale -> Otsu -> closing (``disk_factor``) -> opening (5)."""
    return VipsCompose(
        [
            VipsGrayScaleFilter(),
            VipsOtsu(),
            VipsClosing(disc_factor=disk_factor),
            VipsOpening(disc_factor=5),
        ]
    )
