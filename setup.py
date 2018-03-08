from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='ChimpsTool',
    version='0.0',
    # py_modules=['cmd'],
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'cmd=src.cli:main',
        ],
    },
)
