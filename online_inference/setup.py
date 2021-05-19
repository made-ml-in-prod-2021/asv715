from setuptools import find_packages, setup

setup(
    name="online_inference",
    packages=find_packages(),
    version="0.1.0",
    description="Homework 2",
    author="Sergey Alekseyev",
    install_requires=[
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "pandas==1.1.5",
        "numpy==1.20.0",
        "flask==1.1.2",
        "pydantic==1.8.2",
        "pytest==6.1.2",
        "pytest-cov==2.10.1",
        "requests==2.22.0"
    ],
    license="MIT",
)