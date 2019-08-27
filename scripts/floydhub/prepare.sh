#!/bin/bash

mkdir -p /home/data/dictionary

ln -s /features/* /home/data

if [ -d /character-dict ]; then
  ln -s /character-dict/* /home/data/dictionary/
fi

if [ -d /syllable-dict ]; then
  ln -s /syllable-dict/* /home/data/dictionary/
fi

echo "Finished preparing data directory" \
 && echo "------ /home/data -------" \
 && ls -la /home/data/* \
 && ls -la /home/data/dictionary/*