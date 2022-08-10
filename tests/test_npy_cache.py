from pathlib import Path
import yaml

from cloudpathlib import S3Path

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

    vid_path_str = "s3://drivendata-client-zamba/data/raw/noemie/Taï_cam197_683044_652175_20161223/01090065.AVI"
    vid_path_s3 = S3Path(vid_path_str)
    vid_path_path = Path(vid_path_s3.key)

    expected = Path(
        "data/cache/2d1fee2b1e1f78d06aa08bdea88e7661f927bd81/data/raw/noemie/Taï_cam197_683044_652175_20161223/01090065.npy"
    )

    # test video path as string, S3Path, and Path
    for video_path in [vid_path_str, vid_path_s3, vid_path_path]:
        path = get_cached_array_path(config, video_path, config.cache_dir)
        assert path == expected

    # pass the cache_dir as a Path
    path = get_cached_array_path(config, vid_path_path, Path(config.cache_dir))
    assert isinstance(path, Path)
    assert path == expected

    # pass the cache_dir as S3Path
    cache_dir = S3Path("s3://drivendata-client-zamba") / config.cache_dir
    path = get_cached_array_path(config, vid_path_path, cache_dir)
    expected_s3 = S3Path(
        "s3://drivendata-client-zamba/data/cache/2d1fee2b1e1f78d06aa08bdea88e7661f927bd81/data/raw/noemie/Taï_cam197_683044_652175_20161223/01090065.npy"
    )
    assert isinstance(path, S3Path)
    assert path == expected_s3

    # pass the config as a dictionary
    config_dict = config.dict()
    path = get_cached_array_path(config_dict, vid_path_str, config.cache_dir)
    assert path == expected

    # NOTE: if you use dict(config) and config.dict(), you get two dictionaries
    # that are equal, but their string representations are not, so their digests
    # are different. We should use config.dict() everywhere.

    # changing config.cache_dir and/or config.cleanup_cache should not affect the key
    config_dict = config.dict()
    config_dict["cache_dir"] = "something/else"
    config_dict["cleanup_cache"] = "not even bool"
    path = get_cached_array_path(config_dict, vid_path_path, config.cache_dir)
    assert path == expected

    # changing anything else should
    config_dict["total_frames"] = 8
    path = get_cached_array_path(config_dict, vid_path_path, config.cache_dir)
    expected_different = Path(
        "data/cache/9becb6d6dfe6b9970afe05af06ef49af4881bd73/data/raw/noemie/Taï_cam197_683044_652175_20161223/01090065.npy"
    )
    assert path == expected_different
