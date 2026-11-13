for seed in 0 1 2 3 4; do
  python vqe_task_paralell_Heisenberg.py --seed "$seed"
done

for seed in 0 1 2 3 4; do
  python vqe_task_paralell_TFIM.py --seed "$seed"
done

for seed in 0 1 2 3 4; do
  python vqe_task_paralell_TFCluster.py --seed "$seed"
done

for seed in 0 1 2 3 4;do
  python vqe_task_paralell_TFIM_half_noise.py --seed $seed --noise 'True'
done

for seed in 0 1 2 3 4;do
  python vqe_task_paralell_Heisenberg_half_noise.py --seed $seed --noise 'True'
done

for seed in 0 1 2 3 4;do
  python vqe_task_paralell_TFCluster_half_noise.py --seed $seed --noise 'True'
done
