from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="linear_regression_from_scratch",
    version="0.1.0",
    author="M. Hossein Ghaemi",
    author_email="h.ghaemi.2003@gmail.com",
    description="A simple implementation of linear regression using gradient descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hghaemi/linear_regression_from_scratch.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
    "numpy>=1.19.0",
    "matplotlib>=3.3.0",
    ],
    extras_require={
    "dev": [
        "pytest>=6.0",
        "pytest-cov>=2.0",
        "black>=21.0",
        "flake8>=3.8",
    ],
    },
)