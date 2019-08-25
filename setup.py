from setuptools import setup

with open("./requirements.txt", "r") as f:
    requirements = list(map(lambda x: x.strip(), f.readlines()))

setup(
   name="attacut",
   version="1.0",
   description="A useful module",
   author="Man Foo",
   author_email="foomail@foo.com",
   packages=["attacut"],  #same as name
   install_requires=requirements,
   scripts=["scripts/attacut-cli"]
)