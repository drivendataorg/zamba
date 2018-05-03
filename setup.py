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
    version='0.1',
    description='Zamba is a tool to identify the species seen in camera trap videos from sites in central Africa.',
    author='DrivenData',
    author_email='info@drivendata.org',
    url='https://github.com/drivendataorg/zamba',
    download_url='https://github.com/drivendataorg/zamba/archive/0.1.tar.gz',
    keywords=['deep learning', 'camera', 'africa', 'classifier'],
    install_requires=requirements,
    extras_require={
        "cpu": ["tensorflow==1.7.0"],
        "gpu": ["tensorflow-gpu==1.7.0"]
    },
    entry_points={
        'console_scripts': [
            'zamba=zamba.cli:main',
        ],
    },
)
