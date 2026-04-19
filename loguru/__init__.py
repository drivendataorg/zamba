class _StubLogger:
    def __init__(self):
        self._level = "INFO"

    # Compatibility methods used in the codebase
    def remove(self):
        # No handlers to remove in the stub
        return None

    def add(self, *args, **kwargs):
        # Accept arguments but do nothing; could print for debugging
        return None

    def setLevel(self, level):
        self._level = level

    def debug(self, *args, **kwargs):
        print(*args, **kwargs)

    def info(self, *args, **kwargs):
        print(*args, **kwargs)

    def warning(self, *args, **kwargs):
        print(*args, **kwargs)

    def error(self, *args, **kwargs):
        print(*args, **kwargs)

    def exception(self, *args, **kwargs):
        print(*args, **kwargs)

# Expose a singleton instance matching the loguru API
logger = _StubLogger()
