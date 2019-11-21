from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="attacut",
    version="1.0.6",
    description="Fast and Reasonably Accurate Word Tokenizer for Thai",
    url="https://github.com/PyThaiNLP/attacut",
    author="Pattarawat Chormai et al.",
    author_email="foomail@foo.com",
    packages=["attacut", "attacut.models", "attacut.artifacts"],
    install_requires=[
        "docopt>=0.6.2",
        "fire>=0.1.3",
        "nptyping>=0.2.0",
        "numpy>=1.17.0",
        "pyyaml>=5.1.2",
        "six>=1.12.0",
        "ssg>=0.0.4",
        "torch>=1.2.0",
    ],
    scripts=["scripts/attacut-cli"],
    package_data={"attacut": ["artifacts/**/*"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Natural Language :: Thai",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
