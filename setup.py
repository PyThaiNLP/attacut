from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("./requirements.txt", "r") as f:
    requirements = list(map(lambda x: x.strip(), f.readlines()))

setup(
    name="attacut",
    version="1.0.0",
    description="Yet Another Word Tokenizer for Thai",
    author="Pattarawat Chormai et al.",
    author_email="foomail@foo.com",
    packages=["attacut", "attacut.models", "attacut.artifacts"],
    install_requires=requirements,
    scripts=["scripts/attacut-cli"],
    package_data={"attacut": ["artifacts/**/*"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)