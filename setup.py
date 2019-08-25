from setuptools import setup

with open("./deps.txt", "r") as f:
    dependencies = map(lambda x: x.strip(), f.readlines())

print(dependencies)

setup(
   name="attacut",
   version="1.0",
   description="A useful module",
   author="Man Foo",
   author_email="foomail@foo.com",
   packages=["attacut"],  #same as name
   install_requires=dependencies,
   scripts=["scripts/attacut-cli"]
)