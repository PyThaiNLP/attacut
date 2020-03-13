#!/bin/bash

TAG="$1"

echo "Publishing $TAG to pypi"


DIR="./dist"

if [ -d "$DIR" ]; then
  echo "Cleanup $DIR" && rm -rf $DIR
fi

git checkout $TAG \
    && python setup.py sdist bdist_wheel \
    && twine check dist/* \
    && twine upload dist/* \
    && git checkout - \
    && echo "Please don't forget to push: git push && git push --tags"