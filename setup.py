import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="terrace",
    version="0.0.61",
    author="Michael Brocidiacono",
    author_email="",
    description="high level PyTorch utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mixarcid/terrace",
    packages=setuptools.find_packages(
        where='src',
        include=['*'],
    ),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['torch>=1.6.0']
)
