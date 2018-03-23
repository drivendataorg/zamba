from pathlib import Path
from setuptools import setup


req_path = Path(Path(__file__).parent, 'requirements.txt')
with open(req_path, 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='zamba',
    version='0.0',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'zamba=zamba.cli:main',
        ],
    },
)
