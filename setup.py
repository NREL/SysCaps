import re
import io
import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read()

with open('README.md') as f:
    readme = f.read()

# Read the version from the __init__.py file without importing it
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()
    
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
    
setup(
      name='syscaps',
      version=find_version('syscaps', '__init__.py'),
      description='System captions for surrogates of complex system simulations.',
      author='Patrick Emami, Zhaonan Li, Saumya Sinha, Truc Nguyen',
      author_email='Patrick.Emami@nrel.gov',
      long_description=readme,
      long_description_content_type='text/markdown',
      install_requires=requirements,
      packages=find_packages(include=['syscaps',
                                      'syscaps.data',
                                      'syscaps.evaluation',
                                      'syscaps.models'],
                             exclude=['test']),
      package_data={'syscaps': ['configs/*.toml']},
      license='BSD 3-Clause',
      python_requires='>=3.9',
      keywords=['surrogates', 'energy', 'buildings'],
      classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ]
)
