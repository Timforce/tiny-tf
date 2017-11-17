#!/bin/bash

python ./test_net.py \
--gpu 0 \
--weights ./output/Resnet101_tiny/ \
--cfg ./cfgs/tiny_resnet101.yml \
--refBox ./data/RefBox_N25_scaled.mat \
--pkl_file ./data/pickles/wider_val_detail.pkl \
--network Resnet101_test
