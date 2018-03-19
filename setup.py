from setuptools import setup

with open('requirements.txt', 'r') as f:
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
