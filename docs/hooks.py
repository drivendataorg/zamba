"""mkdocs hooks.

Make ``mkdocstrings`` resilient to third-party packages that register broken
``mkdocstrings.<handler>.templates`` entry points.

``pixeltable-yolox`` (pulled in by zamba's ``video`` extra) ships an entry point
``tool.doc_plugins.mkdocstrings:get_templates_path`` whose ``tool`` package is not
included in its published wheel. When ``mkdocstrings`` initializes its Python
handler it eagerly loads every template extension in that entry point group, so
importing the missing ``tool`` module raises ``ModuleNotFoundError`` and aborts
the entire docs build -- even though it has nothing to do with our docs.

We patch the loader to skip (with a warning) any extension that fails to load,
instead of letting one broken third-party entry point break the build. This only
matters when the ``video`` extra is installed alongside the docs dependencies
(e.g. a combined local dev environment); CI builds docs with ``.[docs]`` only.
"""

import logging
from importlib.metadata import entry_points

from mkdocstrings.handlers.base import BaseHandler

log = logging.getLogger("mkdocs.hooks")


def _safe_get_extended_templates_dirs(self, handler: str):
    dirs = []
    for extension in entry_points(group=f"mkdocstrings.{handler}.templates"):
        try:
            dirs.append(extension.load()())
        except Exception as exc:  # noqa: BLE001 -- never let one bad plugin kill the build
            log.warning(
                "Skipping unloadable mkdocstrings template extension "
                f"{extension.name!r} ({extension.value!r}): {exc}"
            )
    return dirs


# Apply at import time (hooks are imported during config load, before pages render).
BaseHandler.get_extended_templates_dirs = _safe_get_extended_templates_dirs
