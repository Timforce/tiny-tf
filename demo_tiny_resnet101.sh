#!/bin/bash

python ./demo_net.py \
--gpu 0 \
--weights ./output/Resnet101_tiny/ \
--cfg ./cfgs/tiny_resnet101.yml \
--refBox ./data/RefBox_N25_scaled.mat \
--network Resnet101_test
