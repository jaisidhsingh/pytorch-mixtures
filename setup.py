import setuptools


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line and not line.startswith('#')]

setuptools.setup(
    name="pytorch-mixtures",
    version="0.1.0",
    author="Jaisidh Singh",
    author_email="jaisidhsingh@gmail.com",
    description="The one-stop solution to easily integrate MoE & MoD layers into custom PyTorch code.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jaisidhsingh/pytorch-mixtures",
    packages=setuptools.find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
