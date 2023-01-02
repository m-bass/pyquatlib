import pathlib

from setuptools import find_packages, setup

# The parent directory
DIR = pathlib.Path(__file__).parent

# The text of the README file
README = (DIR / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="pyquatlib",
    version="1.0.0",
    description="yet another library for quaternions",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/m-bass/pyquatlib",
    author="Marco Bosatta",
    author_email="mb@xp-obj.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=["numpy", "pandas", "numba"],
    python_requires=">=3",
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main",
        ]
    },
    test_suite="tests",
)
