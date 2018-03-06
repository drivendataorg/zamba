from setuptools import setup


setup(
    name='ChimpsTool',
    version='0.0',
    # py_modules=['cmd'],
    install_requires=[
        'click',
        'ffmpy',
        'numpy',
        'pandas',
        'scikit-learn',
        'sphinx',
        'tensorflow',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'cmd=src.cli:main',
        ],
    },
)
