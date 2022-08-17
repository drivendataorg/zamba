from pathlib import Path
import yaml

from zamba.data.video import (
    VideoLoaderConfig,
    npy_cache,
    get_cached_array_path,
    load_video_frames,
)

config_yaml = """
    cache_dir: local_data/cache
    crop_bottom_pixels: 50
    early_bias: false
    ensure_total_frames: true
    evenly_sample_total_frames: false
    fps: 4.0
    frame_indices: null
    frame_selection_height: null
    frame_selection_width: null
    i_frames: false
    megadetector_lite_config:
        confidence: 0.25
        fill_mode: score_sorted
        image_height: 640
        image_width: 640
        n_frames: 16
        nms_threshold: 0.45
        seed: 55
        sort_by_time: true
    model_input_height: 240
    model_input_width: 426
    pix_fmt: rgb24
    scene_threshold: null
    total_frames: 16
    cleanup_cache: false
    cache_dir: data/cache
"""


def test_get_cached_array_path():
    config_dict = yaml.safe_load(config_yaml)
    config = VideoLoaderConfig(**config_dict)

    # NOTE: the validation in VideoLoaderConfig changes some fields,
    # so dict(config) != config_dict

    cached_load_video_frames = npy_cache(
        cache_path=config.cache_dir, cleanup=config.cleanup_cache
    )(load_video_frames)
    assert isinstance(cached_load_video_frames, type(load_video_frames))

    vid_path_str = "data/raw/noemie/Taï_cam197_683044_652175_20161223/01090065.AVI"
    vid_path = Path(vid_path_str)

    expected_cache_path = vid_path.with_suffix(".npy")
    expected_hash = "2d1fee2b1e1f78d06aa08bdea88e7661f927bd81"
    expected = config.cache_dir / expected_hash / expected_cache_path

    # test video path as string or Path
    for video_path in [vid_path_str, vid_path]:
        path = get_cached_array_path(video_path, config)
        assert path == expected

    # pass the cache_dir as a Path
    config_dict = yaml.safe_load(config_yaml)
    config_dict["cache_dir"] = Path(config_dict["cache_dir"])
    config = VideoLoaderConfig(**config_dict)
    path = get_cached_array_path(vid_path, config)
    assert path == expected

    # changing config.cleanup_cache should not affect the key
    config_dict = yaml.safe_load(config_yaml)
    config_dict["cleanup_cache"] = True
    config = VideoLoaderConfig(**config_dict)
    path = get_cached_array_path(vid_path, config)
    assert path == expected

    # changing config.config_dir should change the path but not the hash
    config_dict = yaml.safe_load(config_yaml)
    config_dict["cache_dir"] = "something/else"
    config = VideoLoaderConfig(**config_dict)
    path = get_cached_array_path(vid_path, config)
    expected_different_path = config.cache_dir / expected_hash / expected_cache_path
    assert path == expected_different_path

    # changing anything else should change the key but not the path
    config_dict = yaml.safe_load(config_yaml)
    config_dict["total_frames"] = 8

    config = VideoLoaderConfig(**config_dict)
    path = get_cached_array_path(vid_path, config)
    different_hash = "9becb6d6dfe6b9970afe05af06ef49af4881bd73"
    expected_different_hash = config.cache_dir / different_hash / expected_cache_path
    assert path == expected_different_hash
