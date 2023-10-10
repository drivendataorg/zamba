import os

import pytest

from pydantic import ValidationError

from zamba.data.video import VideoLoaderConfig
from zamba.models.densepose import DensePoseManager, DensePoseConfig
from zamba.models.densepose.densepose_manager import MODELS

from conftest import ASSETS_DIR


@pytest.fixture
def chimp_video_path():
    return ASSETS_DIR / "densepose_tests" / "chimp.mp4"


@pytest.fixture
def chimp_image_path():
    return ASSETS_DIR / "densepose_tests" / "chimp.jpg"


@pytest.mark.skipif(
    not bool(int(os.environ.get("ZAMBA_RUN_DENSEPOSE_TESTS", 0))),
    reason="""Skip the densepose specific tests unless environment variable \
ZAMBA_RUN_DENSEPOSE_TESTS is set to 1.""",
)
@pytest.mark.parametrize("model", ("animals", "chimps"))
def test_image(model, chimp_image_path, tmp_path):
    dpm = DensePoseManager(model=MODELS[model])

    # segmentation
    image, preds = dpm.predict_image(chimp_image_path)
    assert image.shape == (427, 640, 3)
    assert len(preds) > 0

    # serialize results
    serialized = dpm.serialize_image_output(
        preds, filename=tmp_path / f"output_{model}.json", write_embeddings=False
    )
    deserialized = dpm.deserialize_output(filename=tmp_path / f"output_{model}.json")

    assert serialized is not None
    assert (tmp_path / f"output_{model}.json").stat().st_size > 0
    assert len(deserialized) == len(preds)

    # visualize image
    visualized_img_arr = dpm.visualize_image(
        image, preds, output_path=(tmp_path / f"viz_image_{model}.jpg")
    )
    assert (tmp_path / f"viz_image_{model}.jpg").stat().st_size > 0
    assert visualized_img_arr.shape == image.shape
    assert (visualized_img_arr != image.shape).any()

    # anantomy
    if model == "chimps":
        anatomy_info = dpm.anatomize_image(
            visualized_img_arr, preds, output_path=(tmp_path / f"anatomized_{model}.csv")
        )

        # output to disk
        assert anatomy_info.shape in [
            (2, 44),
            (1, 44),
        ]  # depends on number of chimps identified; varies by version
        assert (anatomy_info > 0).any().any()
        assert (tmp_path / f"anatomized_{model}.csv").stat().st_size > 0


@pytest.mark.skipif(
    not bool(int(os.environ.get("ZAMBA_RUN_DENSEPOSE_TESTS", 0))),
    reason="""Skip the densepose specific tests unless environment variable \
ZAMBA_RUN_DENSEPOSE_TESTS is set to 1.""",
)
@pytest.mark.parametrize("model", ("animals", "chimps"))
def test_video(model, chimp_video_path, tmp_path):
    dpm = DensePoseManager(model=MODELS[model])

    # segmentation
    vid, preds = dpm.predict_video(
        chimp_video_path, video_loader_config=VideoLoaderConfig(fps=0.2)
    )
    assert vid.shape == (3, 180, 320, 3)
    assert len(preds) > 0

    # serialize results
    serialized = dpm.serialize_video_output(
        preds, filename=tmp_path / f"output_{model}.json", write_embeddings=False
    )
    deserialized = dpm.deserialize_output(filename=tmp_path / f"output_{model}.json")

    assert serialized is not None
    assert (tmp_path / f"output_{model}.json").stat().st_size > 0
    assert len(deserialized) == len(preds)

    # visualize image
    visualized_vid_arr = dpm.visualize_video(
        vid, preds, output_path=(tmp_path / f"viz_vid_{model}.mp4")
    )
    assert (tmp_path / f"viz_vid_{model}.mp4").stat().st_size > 0
    assert visualized_vid_arr.shape == vid.shape
    assert (visualized_vid_arr != vid).any()

    # anantomy
    if model == "chimps":
        anatomy_info = dpm.anatomize_video(
            visualized_vid_arr, preds, output_path=(tmp_path / f"anatomized_{model}.csv")
        )

        # output to disk
        assert anatomy_info.shape[0] in [
            8,
            9,
            10,
        ]  # depends on number of chimps identified; varies by version
        assert anatomy_info.shape[1] == 46
        assert (anatomy_info > 0).any().any()
        assert (tmp_path / f"anatomized_{model}.csv").stat().st_size > 0


@pytest.mark.skipif(
    not bool(int(os.environ.get("ZAMBA_RUN_DENSEPOSE_TESTS", 0))),
    reason="""Skip the densepose specific tests unless environment variable \
ZAMBA_RUN_DENSEPOSE_TESTS is set to 1.""",
)
@pytest.mark.parametrize("model", ("animals", "chimps"))
def test_denseposeconfig(model, tmp_path):
    # validation failures
    with pytest.raises(ValidationError):
        DensePoseConfig(
            video_loader_config=VideoLoaderConfig(fps=0.2),
            output_type="bananas",
            render_output=True,
            embeddings_in_json=False,
            data_dir=ASSETS_DIR / "densepose_tests",
            save_dir=tmp_path,
        )

    dpc = DensePoseConfig(
        video_loader_config=VideoLoaderConfig(fps=0.2),
        output_type="segmentation" if model == "animals" else "chimp_anatomy",
        render_output=True,
        embeddings_in_json=False,
        data_dir=ASSETS_DIR / "densepose_tests",
        save_dir=tmp_path,
    )

    dpc.run_model()

    # ensure all outputs are saved in save_dir
    assert (tmp_path / "chimp_denspose_video.mp4").exists()
    assert (tmp_path / "chimp_denspose_labels.json").exists()

    if model == "chimp_anatomy":
        assert (tmp_path / "chimp_denspose_anatomy.csv").exists()
