from pathlib import Path
from setuptools import setup


req_path = Path(Path(__file__).parent, 'requirements.txt')
with open(req_path, 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='djamba',
    version='0.0',
    # py_modules=['djamba'],
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'djamba=djamba.cli:main',
        ],
    },
)
