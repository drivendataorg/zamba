"""Smoke tests for optional image/video extra isolation.

These tests are unmarked so they run in both ``make test-image-only`` and
``make test-video-only`` (see pytest marker expression in the Makefile).

Marked ``@pytest.mark.image`` / ``@pytest.mark.video`` tests exercise a real
model end-to-end in the corresponding isolation venv.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ASSETS_DIR, TEST_VIDEOS_DIR


def _zamba_executable() -> list[str]:
    """Use the zamba CLI from the same venv as the running interpreter."""
    venv_zamba = Path(sys.executable).with_name("zamba")
    if venv_zamba.exists():
        return [str(venv_zamba)]
    found = shutil.which("zamba")
    assert found is not None
    return [found]


def test_zamba_cli_imports():
    from zamba.cli import app

    assert app is not None


def test_image_subcommand_when_image_extra_installed():
    pytest.importorskip("megadetector")

    result = subprocess.run(
        _zamba_executable() + ["image", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_video_predict_help():
    result = subprocess.run(
        _zamba_executable() + ["predict", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_image_cli_unavailable_without_image_extra():
    try:
        import megadetector  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("image extra is installed")

    result = subprocess.run(
        _zamba_executable() + ["image", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.image
def test_image_model_predict_smoke(tmp_path):
    """Image-only install: run speciesnet inference on a single test image."""
    pytest.importorskip("megadetector")

    data_dir = tmp_path / "images"
    data_dir.mkdir()
    shutil.copy(ASSETS_DIR / "images" / "small_cat.jpg", data_dir)
    save_dir = tmp_path / "out"

    result = subprocess.run(
        _zamba_executable()
        + [
            "image",
            "predict",
            "--data-dir",
            str(data_dir),
            "--model",
            "speciesnet",
            "--save-dir",
            str(save_dir),
            "--yes",
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert (save_dir / "zamba_predictions.csv").exists()


@pytest.mark.video
def test_video_model_predict_smoke(tmp_path):
    """Video-only install: run time_distributed inference (dry-run) on one clip."""
    pytest.importorskip("ffmpeg")

    video_src = TEST_VIDEOS_DIR / "data" / "raw" / "benjamin" / "04250002.MP4"
    if not video_src.exists():
        pytest.skip(f"test video not found: {video_src}")

    data_dir = tmp_path / "videos"
    data_dir.mkdir()
    shutil.copy(video_src, data_dir)

    result = subprocess.run(
        _zamba_executable()
        + [
            "predict",
            "--data-dir",
            str(data_dir),
            "--config",
            str(ASSETS_DIR / "sample_predict_config.yaml"),
            "--dry-run",
            "--yes",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr or result.stdout
