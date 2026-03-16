"""Regression tests for download.validate_dataset."""

from pathlib import Path

from download import validate_dataset


def _create_pair(tmp: Path, mouse: str, name: str) -> None:
    """Create an image/mask pair with the dataset naming convention."""
    img_dir = tmp / mouse / "Images"
    mask_dir = tmp / mouse / "Masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{name}.png").write_bytes(b"\x89PNG")
    (mask_dir / f"{name}_mask.png").write_bytes(b"\x89PNG")


def test_validate_matching_pairs(tmp_path):
    """Image/mask pairs with _mask suffix should match correctly."""
    _create_pair(tmp_path, "M01", "image_001")
    _create_pair(tmp_path, "M01", "image_002")
    _create_pair(tmp_path, "M02", "image_003")

    result = validate_dataset(tmp_path)
    assert result["n_images"] == 3
    assert result["n_masks"] == 3
    assert result["match"] is True
    assert result["missing_masks"] == 0
    assert result["missing_images"] == 0


def test_validate_missing_mask(tmp_path):
    """An image with no corresponding mask should be detected."""
    _create_pair(tmp_path, "M01", "image_001")

    # Add an extra image with no mask
    img_dir = tmp_path / "M01" / "Images"
    (img_dir / "image_002.png").write_bytes(b"\x89PNG")

    result = validate_dataset(tmp_path)
    assert result["match"] is False
    assert result["missing_masks"] == 1


def test_validate_empty_directory(tmp_path):
    """An empty directory should report no match."""
    result = validate_dataset(tmp_path)
    assert result["match"] is False
    assert result["n_images"] == 0
    assert result["n_masks"] == 0
