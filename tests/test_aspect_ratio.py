"""Tests for the `size` "WxH" → MiniMax `aspect_ratio` mapping."""

from __future__ import annotations

import pytest

from owui_minimax_image_proxy.models import SUPPORTED_ASPECT_RATIOS, size_to_aspect_ratio


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1024x1024", "1:1"),
        ("1792x1024", "16:9"),
        ("1024x1792", "9:16"),
        ("1344x768", "16:9"),  # 1.75 → nearest of 16:9 (1.778) vs 21:9 (2.333)
        ("1152x864", "4:3"),
        ("1536x1024", "3:2"),
        ("1024x1536", "2:3"),
        ("864x1152", "3:4"),
        ("2560x1080", "21:9"),
        # Odd / non-standard sizes → nearest supported ratio.
        ("1000x1000", "1:1"),
        ("1000x500", "16:9"),  # ratio 2.0 → 16:9 (1.778) beats 21:9 (2.333)
        ("500x1000", "9:16"),  # ratio 0.5 → 9:16 (0.5625) beats 2:3 (0.667)
        ("1234x777", "3:2"),  # ratio 1.588 → 3:2 (1.5) beats 16:9 (1.778)
        ("777x1234", "2:3"),  # ratio 0.630 → 2:3 (0.667) beats 9:16 (0.5625)
        ("512x512", "1:1"),
        ("2048x2048", "1:1"),
        ("100x2100", "9:16"),  # extreme portrait
        ("2100x100", "21:9"),  # extreme landscape
        (" 1024 x 1024 ", "1:1"),  # whitespace tolerated
        ("1024X1024", "1:1"),  # uppercase X tolerated
    ],
)
def test_size_maps_to_nearest_aspect_ratio(size: str, expected: str) -> None:
    assert size_to_aspect_ratio(size) == expected


@pytest.mark.parametrize(
    "size",
    [
        "",
        "1024",
        "1024x",
        "x1024",
        "1024*1024",
        "1024x1024x1024",
        "-1024x1024",
        "abc",
    ],
)
def test_malformed_size_raises(size: str) -> None:
    with pytest.raises(ValueError, match="WxH"):
        size_to_aspect_ratio(size)


@pytest.mark.parametrize("size", ["0x1024", "1024x0"])
def test_zero_dimension_size_raises(size: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        size_to_aspect_ratio(size)


def test_every_supported_ratio_is_its_own_nearest() -> None:
    """Each supported ratio string, parsed as WxH, must map back to itself."""
    for label, w, h in SUPPORTED_ASPECT_RATIOS:
        assert size_to_aspect_ratio(f"{w * 64}x{h * 64}") == label
