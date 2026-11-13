#!/bin/bash
# 循环运行 VqeTrainerLiH.py，种子从 0 到 4

for seed in 0 1 2 3 4;
do
    echo "正在运行，种子为 $seed"
    python vqe_task_paralell_Heisenberg_half_noise.py --seed $seed --noise True
done

for seed in 0 1 2 3 4;
do
    echo "正在运行，种子为 $seed"
    python vqe_task_paralell_TFIM_half_noise.py --seed $seed --noise True
done

for seed in 0 1 2 3 4;
do
    echo "正在运行，种子为 $seed"
    python VqeTrainer_TFCluster_half_noise.py --seed $seed --noise True
done
