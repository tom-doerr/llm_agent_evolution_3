from setuptools import setup, find_packages

setup(
    name="evolver",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "dspy-ai",
        "toml",
        "rich",
    ],
    entry_points={
        'console_scripts': [
            'evolver=evolver.main:main',
        ],
    },
)
