from setuptools import setup, find_packages

setup(
    name='ckmeans',
    packages=find_packages(),
    description='A consensus K-Means implementation.',
    author='Tankred Ott',
    platforms=['any'],
    python_requires='>=3.6',
)
