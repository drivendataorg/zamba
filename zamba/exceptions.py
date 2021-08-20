class ZambaFfmpegException(Exception):
    def __init__(self, stderr: bytes):
        super().__init__(
            f"Video loading failer with error:\n{stderr.decode('utf8', errors='replace')}"
        )
