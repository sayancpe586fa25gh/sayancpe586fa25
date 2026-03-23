#!/bin/bash
python imagenet_impl.py --epochs 10000 --train_ratio 0.01 --val_ratio 0.003 > imagenet_res_final_3.log
