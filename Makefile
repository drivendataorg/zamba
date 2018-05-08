test:
	python -m pytest -s zamba

# requirements for local OSX development
# requires `brew install gcc`
reqs:
	pip install -U pip Cython
	env CC=gcc-5 CXX=g++-5 pip install -r requirements.txt

clean_pycache:
	find . -name *.pyc -delete && find . -name __pycache__ -delete

ci_env_vars:
	jet encrypt --key-path drivendataorg_chimps-tool.aes .env env.encrypted

clean: clean_pycache
	rm -rf dist
	rm -rf zamba.egg-info

build: clean
	python setup.py sdist

distribute_testpypi: build
	twine upload --repository pypitest dist/*.tar.gz

distribute_pypi: build
	twine upload --repository pypitest dist/*.tar.gz

