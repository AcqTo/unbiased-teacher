#!/bin/bash

python ../cocosplit//cocosplit.py ../datasets/instagram/result.json train.json val.json -s 0.8
python ../cocosplit/cocosplit.py train.json train90.json train10.json -s 0.9

mv ../cocosplit/train.json  ../datasets/instagram/annotations/
mv ../cocosplit/val.json  ../datasets/instagram/annotations/