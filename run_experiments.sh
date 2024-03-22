#!/bin/bash

# ./run_experiments.sh

# Script to reproduce results

for ((i=1;i<=5;i+=1))
do 
    nohup python run.py --env Reacher-v2 --policy SIR3 --isSparse --reLabeling --seed $i > Reacher$i.txt 2>&1 &
    nohup python run.py --env Pusher-v2 --policy SIR3 --isSparse --reLabeling --seed $i > Pusher$i.txt 2>&1 &
    nohup python run.py --env HalfCheetah-v2 --policy SIR3 --isSparse --reLabeling --reLabelingDone --seed $i > HalfCheetah$i.txt 2>&1 &
    nohup python run.py --env Hopper-v2 --policy SIR3 --isSparse --reLabeling --reLabelingDone --seed $i > Hopper$i.txt 2>&1 &
    sleep 2h
done
