from typing import Union


class ZambaFfmpegException(Exception):
    def __init__(self, stderr: Union[bytes, str]):
        message = stderr.decode("utf8", errors="replace") if isinstance(stderr, bytes) else stderr
        super().__init__(f"Video loading failer with error:\n{message}")
