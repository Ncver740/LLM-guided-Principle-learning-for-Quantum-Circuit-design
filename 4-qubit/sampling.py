import os.path

import utils
from circuit_sampler import CircuitSampler
import config
import time
import random
import numpy as np
import argparse
import sys
from circuit_sampler_all2all import CircuitSamplerAll2All

def sample_search_circuits(device_name, task, run_id, num_qubit, num_layer, num_sample):
    sample_circuit_process(device_name, task, num_layer, num_sample, num_qubit, run_id)

def sample_circuit_process(device_name, task, num_layer, num_sample, num_qubit, run_id):
    exp_dir_train = f'cir_sample/{device_name}_{task}/training/run_{run_id}/'
    exp_dir_search_pool = f'cir_sample/{device_name}_{task}/search_pool/run_{run_id}/'

    if not os.path.exists(exp_dir_train):
        os.makedirs(exp_dir_train)

    if not os.path.exists(exp_dir_search_pool):
        os.makedirs(exp_dir_search_pool)

    if 'full' in device_name:
        print("使用全连接采样器")
        sampler = CircuitSamplerAll2All(device_name, num_layer, num_qubit)
    else:
        sampler = CircuitSampler(device_name, num_layer, num_qubit)
    samples, qubit_mapping_lst, edges_mapping_list = sampler.sample(num_sample)

    utils.save_pkl(samples[:10000], f'{exp_dir_train}samples.pkl')
    utils.save_pkl(qubit_mapping_lst[:10000], f'{exp_dir_train}qubit_mapping.pkl')
    utils.save_pkl(edges_mapping_list[:10000], f'{exp_dir_train}edges_mapping.pkl')

    utils.save_pkl(samples[10000:], f'{exp_dir_search_pool}samples.pkl')
    utils.save_pkl(qubit_mapping_lst[10000:], f'{exp_dir_search_pool}qubit_mapping.pkl')
    utils.save_pkl(edges_mapping_list[10000:], f'{exp_dir_search_pool}edges_mapping.pkl')


    encoding = []
    for sample in samples:
        adj_matrix, feat_matrix = utils.circuit_list_to_adj(circuit=sample[0], num_qubit=num_qubit,
            gate_type_list=config.device[device_name]['gate_set'], require_feat_matrix=True)
        encoding.append([adj_matrix, feat_matrix])
    utils.save_pkl(encoding[:10000], f'{exp_dir_train}adj_feat_matrix.pkl')
    utils.save_pkl(encoding[10000:], f'{exp_dir_search_pool}adj_feat_matrix.pkl')


if __name__ == '__main__':

    # args = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--sample_num", type=int, default=1000, help="random seed")
    parser.add_argument('--device_name', type=str, default='full_4q', help='')
    parser.add_argument('--task', type=str, default='Heisenberg_4', help='')
    parser.add_argument('--num_layer', type=int, default=10, help='')
    args = parser.parse_args()

    run_id = args.seed
    device_name = args.device_name
    task = args.task
    num_layer = args.num_layer

    print(f"device_name: {device_name}, task: {task}, num_layer: {num_layer}, run_id: {run_id}, sample_num: {args.sample_num}")

    start_time = time.time()
    random.seed(run_id)
    np.random.seed(run_id)
    num_qubit = config.task_configs[args.task]['qubit']
    print(f"num_qubit: {num_qubit}")

    sample_search_circuits(device_name, task, run_id, num_qubit, num_layer, num_sample=args.sample_num)

    end_time = time.time()
    duration = end_time - start_time
    print(f"run time: {int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes")






