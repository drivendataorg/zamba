test:
	python -m pytest -s zamba

# requirements for local OSX development
# requires `brew install gcc`
reqs:
	pip install -U pip Cython
	env CC=gcc-5 CXX=g++-5 pip install -r requirements.txt

clean_pycache:
	find . -name *.pyc -delete && find . -name __pycache__ -delete
