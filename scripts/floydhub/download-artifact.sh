#!/bin/bash

USER="pattt"
LOCAL_DIR="local-artifacts"
DIR="artifact"
FLOYDHUB_PATH="$USER/projects/$1"

CURR_DIR=`pwd`

JOB=`echo $1 | sed "s/\//-/"`
DEST="./$LOCAL_DIR/$JOB"
echo "Downloading articats from $FLOYDHUB_PATH to $DEST"

mkdir $DEST \
    && cd $DEST \
    && floyd data clone -p $DIR $FLOYDHUB_PATH
