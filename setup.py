from pathlib import Path
from setuptools import setup, find_packages


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

long_description = open(Path(__file__).parent / 'README.md').read()

setup(
    name='zamba',
    version='0.1.5',
    description='Zamba is a tool to identify the species seen in camera trap videos from sites in central Africa.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DrivenData',
    author_email='info@drivendata.org',
    url='http://zamba.drivendata.org',
    project_urls={
        'Homepage': 'http://zamba.drivendata.org',
        'Documentation': 'http://zamba.drivendata.org/docs/',
        'Source Code': 'https://github.com/drivendataorg/zamba',
        'DrivenData': 'http://drivendata.co'
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Multimedia :: Video',
    ],
    download_url='https://github.com/drivendataorg/zamba/archive/0.1.tar.gz',
    keywords=['deep learning', 'camera', 'africa', 'classifier'],
    python_requires='>=3.5',
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
    packages=find_packages(exclude=['dist', 'docs', 'zamba.tests']),
)
