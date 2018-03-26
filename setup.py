from pathlib import Path
from setuptools import setup


def load_reqs(p):
    requirements = []
    with open(p, 'r') as f:
        for l in f.readlines():
            if l.startswith('-r'):
                requirements += load_reqs(l.split(' ')[1].strip())
            else:
                r = l.strip()

                if r and not r.startswith('#'):
                    requirements.append(r)
    return requirements


req_path = Path(Path(__file__).parent, 'requirements.txt')
requirements = load_reqs(req_path)

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
