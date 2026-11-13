#!/bin/bash
# 循环运行 VqeTrainerLiH.py，种子从 0 到 4

# ip 115 主机运行
for seed in 0 1 2 3 4;
do
    echo "正在运行，种子为 $seed"
    python VqeTrainer_TFCluster.py --seed $seed --noise True
done

# ip 102 主机运行
for seed in 4 3 2 1 0;
do
    echo "正在运行，种子为 $seed"
    python VqeTrainer_TFCluster.py --seed $seed --noise True
done