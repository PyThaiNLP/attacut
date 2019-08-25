from setuptools import setup

with open("./requirements.txt", "r") as f:
    requirements = list(map(lambda x: x.strip(), f.readlines()))

setup(
    name="attacut",
    version="0.0.2-dev",
    description="Yet Another Tokenizer for Thai",
    author="Man Foo",
    author_email="foomail@foo.com",
    packages=["attacut", "attacut.models", "attacut.artifacts"],
    install_requires=requirements,
    scripts=["scripts/attacut-cli"],
    package_data={"attacut": ["artifacts/**/*"]},
)