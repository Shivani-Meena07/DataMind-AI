"""Setup script for DataMind AI."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="datamind-ai",
    version="1.0.0",
    author="DataMind AI Team",
    author_email="team@datamind.ai",
    description="An AI system that automatically generates database user manuals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/datamind-ai/datamind",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Database",
        "Topic :: Documentation",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "datamind=datamind.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "datamind": ["templates/*.jinja2"],
    },
)
