#!/bin/bash

KEYWORD="hw02"

for i in {1..5}
do
    echo "Run $i"
    python multiclass_impl.py \
        --data data/Android_Malware.csv \
        --lr 0.001 \
        --epochs 100000 \
        --keyword $KEYWORD
done

python multiclass_eval.py --keyword $KEYWORD

