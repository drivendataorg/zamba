from loguru import logger

from cloudpathlib import S3Path

from zamba.models.config import WEIGHT_LOOKUP


def upload_models():
    for model in WEIGHT_LOOKUP.keys():
        for bucket in ["", "-eu", "-asia"]:
            origin = S3Path(WEIGHT_LOOKUP[model]["private_weights"])
            public_file_name = WEIGHT_LOOKUP[model]["public_weights"]
            destination = S3Path(f"s3://drivendata-public-assets{bucket}/{public_file_name}")

            logger.info(f"Uploading {origin} to {destination}")
            origin.copy(destination, force_overwrite_to_cloud=True)


if __name__ == "__main__":
    upload_models()
