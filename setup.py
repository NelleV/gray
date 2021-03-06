# Copyright (C) 2018 Łukasz Langa
from setuptools import setup
import sys
import versioneer

assert sys.version_info >= (3, 6, 0), "gray requires Python 3.6+"
from pathlib import Path  # noqa E402

CURRENT_DIR = Path(__file__).parent


def get_long_description() -> str:
    readme_md = CURRENT_DIR / "README.md"
    with open(readme_md, encoding="utf8") as ld_file:
        return ld_file.read()


setup(
    name="gray",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="The uncompromising code formatter.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords="automation formatter yapf autopep8 pyfmt gofmt rustfmt",
    author="Łukasz Langa",
    author_email="lukasz@langa.pl",
    url="https://github.com/psf/gray",
    license="MIT",
    py_modules=["gray", "gray.grayd", "_version", "gray.externals.blib2to3"],
    packages=["gray.externals.blib2to3.pgen2"],
    package_data={"gray": ["*.txt"]},
    python_requires=">=3.6",
    zip_safe=False,
    install_requires=[
        "click>=6.5",
        "attrs>=18.1.0",
        "appdirs",
        "toml>=0.9.4",
        "typed-ast>=1.3.1",
    ],
    extras_require={"d": ["aiohttp>=3.3.2", "aiohttp-cors"]},
    test_suite="gray.tests.test_gray",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
    ],
    entry_points={
        "console_scripts": ["gray=gray:patched_main", "grayd=grayd:patched_main [d]"]
    },
)
