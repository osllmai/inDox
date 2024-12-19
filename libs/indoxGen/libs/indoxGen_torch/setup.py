from setuptools import setup, find_packages

# Read the requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Extract package names
packages = [req.split("==")[0] for req in requirements if "==" in req]

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="indoxGen-torch",
    version="0.1.0",
    license="AGPL-3.0-or-later",
    packages=find_packages(),
    include_package_data=True,
    description="Indox Synthetic Data Generation (GAN-pytorch)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="nerdstudio",
    author_email="ashkan@nematifamilyfundation.onmicrosoft.com",
    url="https://github.com/osllmai/inDox/libs/IndoxGen/tree/master/libs/indoxGen_torch",
    keywords=[
        "AI",
        "deep learning",
        "language models",
        "synthetic data generation",
        "machine learning",
        "NLP",
    ],
    install_requires=[
        "torch==2.4.1",
        "seaborn==0.11.1",
        "matplotlib==3.4.2",
        "dython==0.6.4.post1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
)
