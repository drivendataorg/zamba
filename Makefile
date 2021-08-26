.PHONY: docs docs-serve

## Build the static version of the docs
docs:
	cd docs && mkdocs build

## Serve documentation to livereload while you work on them
docs-serve:
	cd docs && mkdocs serve