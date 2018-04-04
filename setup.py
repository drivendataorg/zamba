from pathlib import Path
from setuptools import setup


def load_reqs(path):
    reqs = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('-r'):
                reqs += load_reqs(line.split(' ')[1].strip())
            else:
                req = line.strip()
                if req and not req.startswith('#'):
                    reqs.append(req)
    return reqs


req_path = Path(__file__).parent / 'requirements.txt'
requirements = load_reqs(req_path)

setup(
    name='zamba',
    version='0.0',
    install_requires=requirements,
    extras_require={
        "tf": ["tensorflow==1.5.0"],
        "tf_gpu": ["tensorflow-gpu==1.4.1"]
    },
    entry_points={
        'console_scripts': [
            'zamba=zamba.cli:main',
        ],
    },
)
