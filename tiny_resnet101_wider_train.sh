#!/bin/bash

python ./train_net.py \
--gpu 0 \
--epochs 50 \
--weights ./data/pretrain_model/Resnet101.npy \
--cfg ./cfgs/tiny_resnet101.yml \
--network Resnet101_train \
--pkl_file ./data/pickles/wider_train_roidb_detail.pkl \
--refBox ./data/RefBox_N25_scaled.mat \
--restore 0
