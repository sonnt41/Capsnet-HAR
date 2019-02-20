#!/usr/bin/env bash

source activate tensorflow 
EXP_DIR=output/lstm

mkdir -r $EXP_DIR

MODEL=$1

if [ $MODEL = 'capsnet' ] ; then
    python ./train.py -model CapsuleNet \
               -unit 100 \
               -r 1 \
               -window_size 100 \
               -epoch 50 \
               -optim adam \
               -lr 0.0003 \
               -batch 32 \
               -gpu 0 \
               -skip 20 \
               -worker 4 \
               -kernel_size 5 \
               -lam_recon 0.0001

fi



if [ $MODEL = 'lstm' ] ; then
    python ./train.py -model DeepConvLSTM \
               -unit 256 \
               -layer 3 \
               -window_size 100 \
               -tpoint 20 \
               -optim adam \
               -lr 0.0003 \
               -batch 64 \
               -epoch 50 \
               -gpu 0 \
               -skip 70 \
               -worker 4 \
               -dropout 0.2 \
               -folding 10
               
fi

if [ $MODEL = 'gru' ] ; then
    python ./train.py -model StackedGRU \
               -unit 100 \
               -layer 4 \
               -window_size 100 \
               -tpoint 20 \
               -optim nesterov \
               -lr 0.00003 \
               -batch 128 \
               -epoch 50 \
               -gpu 0 \
               -skip 60 \
               -worker 4 \
               -dropout 0.2
fi


if [ $MODEL = 'deepsense' ] ; then
    python ./train.py -model DeepSense \
               -unit 100 \
               -layer 4 \
               -window_size 100 \
               -tpoint 20 \
               -optim nesterov \
               -lr 0.001 \
               -batch 32 \
               -epoch 50 \
               -gpu 0 \
               -skip 60 \
               -worker 4 \
               -dropout 0.2
fi