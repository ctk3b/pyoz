#!/bin/bash
# this script will run the full testing of pyOZ

for P in `find . -maxdepth 1 -type d`
do
  cd $P
  pyoz -i picard.in -n picard -o picard.log
  pyoz -i nrcg.in -n nrcg -o nrcg.log
  cd ..
done
