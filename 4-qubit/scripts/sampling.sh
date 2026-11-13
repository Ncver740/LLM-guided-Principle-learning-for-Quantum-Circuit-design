#
#for seed in 0 1 2 3 4;do
#  python sampling.py --seed $seed --task 'TFIM_4' --sample_num 110000 --device_name 'full_4q' --num_layer 10
#done
#
#for seed in 0 1 2 3 4;do
#  python sampling.py --seed $seed --task 'Heisenberg_4' --sample_num 110000 --device_name 'full_4q' --num_layer 10
#done

for seed in 0 1 2 3 4;do
  python sampling.py --seed $seed --task 'TFCluster_4' --sample_num 110000 --device_name 'full_4q' --num_layer 10
done



