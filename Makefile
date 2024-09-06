.PHONY: docs docs-serve clean lint requirements sync_data_down sync_data_up tests

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = zamba
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

ifeq (, $(shell which nvidia-smi))
CPU_OR_GPU ?= cpu
else
CPU_OR_GPU ?= gpu
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
ifeq (${CPU_OR_GPU}, gpu)
	conda install -y cudatoolkit=11.0.3 cudnn=8.0 -c conda-forge
endif
	$(PYTHON_INTERPRETER) -m pip install -U pip torch
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	find . -name ".DS_Store" -type f -delete  # breaks tests on MacOS
	rm -fr .tox/
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist: clean ## builds source and wheel package
	python -m build
	ls -l dist

## Format using black
format:
	black zamba tests

## Lint using flake8 + black
lint:
	flake8 zamba tests
	black --check zamba tests

## Generate assets and run tests
tests: clean-test
	pytest tests -vv

## Run the tests that are just for densepose
densepose-tests:
	ZAMBA_RUN_DENSEPOSE_TESTS=1 pytest tests/test_densepose.py tests/test_cli.py::test_densepose_cli_options -vv

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Generate the index.md page of the docs automatically from README.md
docs-setup:
	sed 's|https://user-images.githubusercontent.com/46792169/138346340-98ee196a-5ecd-4753-b9df-380528091f9e.mp4| \
	<div class="embed-responsive embed-responsive-16by9" width=500> \
    <iframe width=600 height=340 class="embed-responsive-item" src="https://s3.amazonaws.com/drivendata-public-assets/monkey-vid.mp4" \
	frameborder="0" allowfullscreen=""></iframe></div>|g' README.md \
	| sed 's|Visit https://zamba.drivendata.org/docs/ for full documentation and tutorials.||g' \
	| sed 's|https://user-images.githubusercontent.com /46792169/137787221-de590183-042e-4d30-b32b-1d1c2cc96589.mov| \
	<script id="asciicast-1mXKsDiPzgyAZwk8CbdkrG2ac" src="https://asciinema.org/a/1mXKsDiPzgyAZwk8CbdkrG2ac.js" async data-autoplay="true" data-loop=1></script>|g' \
	| sed 's|https://zamba.drivendata.org/docs/stable/||g' \
	> docs/docs/index.md

	sed 's|https://zamba.drivendata.org/docs/stable/|../|g' HISTORY.md > docs/docs/changelog/index.md

## Build the static version of the docs
docs: docs-setup
	cd docs && mkdocs build

## Serve documentation to livereload while you work on them
docs-serve: docs-setup
	cd docs && mkdocs serve

publish_models:
	python -m zamba.models.publish_models

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
