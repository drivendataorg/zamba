from loguru import logger

from cloudpathlib import S3Path

from zamba.models.config import MODEL_MAPPING


def upload_models():
    for model in MODEL_MAPPING.keys():
        for bucket in ["", "-eu", "-asia"]:
            origin = S3Path(MODEL_MAPPING[model]["private_weights"])
            public_file_name = MODEL_MAPPING[model]["public_weights"]
            destination = S3Path(f"s3://drivendata-public-assets{bucket}/{public_file_name}")

            logger.info(f"Uploading {origin} to {destination}")
            origin.copy(destination, force_overwrite_to_cloud=True)


if __name__ == "__main__":
    upload_models()
