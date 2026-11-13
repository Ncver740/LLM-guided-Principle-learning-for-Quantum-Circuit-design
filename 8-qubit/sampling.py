import os.path

import utils
from circuit_sampler import CircuitSampler
import config
import time
import random
import numpy as np
import argparse
import sys

# def sample_pretrain_circuits(device_name, run_id, num_qubit, num_layer, num_sample):
#     exp_dir = f'result/cir_sample/{device_name}/pretrain/run_{run_id}/'
#     sample_circuit_process(device_name, num_layer, num_sample, exp_dir, num_qubit)

# def sample_finetune_circuits(device_name, run_id, num_qubit, num_layer, num_sample):
#     exp_dir = f'result/cir_sample/{device_name}/finetune/run_{run_id}/'
#     sample_circuit_process(device_name, num_layer, num_sample, exp_dir, num_qubit)
#     sample_opt_cfg(num_sample, exp_dir)

def sample_search_circuits(device_name, task, run_id, num_qubit, num_layer, num_sample):
    sample_circuit_process(device_name, task, num_layer, num_sample, num_qubit, run_id)

def sample_circuit_process(device_name, task, num_layer, num_sample, num_qubit, run_id):
    exp_dir_train = f'result/cir_sample/{device_name}_{task}/training/run_{run_id}/'
    exp_dir_search_pool = f'result/cir_sample/{device_name}_{task}/search_pool/run_{run_id}/'
    exp_dir_pretrain = f'result/cir_sample/{device_name}_{task}/pretrain/run_{run_id}/'

    if not os.path.exists(exp_dir_train):
        os.makedirs(exp_dir_train)

    if not os.path.exists(exp_dir_search_pool):
        os.makedirs(exp_dir_search_pool)

    if not os.path.exists(exp_dir_pretrain):
        os.makedirs(exp_dir_pretrain)

    sampler = CircuitSampler(device_name, num_layer, num_qubit)
    samples, qubit_mapping_lst, edges_mapping_list = sampler.sample(num_sample)

    utils.save_pkl(samples[:1000], f'{exp_dir_train}samples.pkl')
    utils.save_pkl(qubit_mapping_lst[:1000], f'{exp_dir_train}qubit_mapping.pkl')
    utils.save_pkl(edges_mapping_list[:1000], f'{exp_dir_train}edges_mapping.pkl')

    utils.save_pkl(samples[1000:101000], f'{exp_dir_search_pool}samples.pkl')
    utils.save_pkl(qubit_mapping_lst[1000:101000], f'{exp_dir_search_pool}qubit_mapping.pkl')
    utils.save_pkl(edges_mapping_list[1000:101000], f'{exp_dir_search_pool}edges_mapping.pkl')

    utils.save_pkl(samples[101000:], f'{exp_dir_pretrain}samples.pkl')
    utils.save_pkl(qubit_mapping_lst[101000:], f'{exp_dir_pretrain}qubit_mapping.pkl')
    utils.save_pkl(edges_mapping_list[101000:], f'{exp_dir_pretrain}edges_mapping.pkl')

    encoding = []
    for sample in samples:
        adj_matrix, feat_matrix = utils.circuit_list_to_adj(circuit=sample[0], num_qubit=num_qubit,
            gate_type_list=config.device[device_name]['gate_set'], require_feat_matrix=True)
        encoding.append([adj_matrix, feat_matrix])
    utils.save_pkl(encoding[:1000], f'{exp_dir_train}adj_feat_matrix.pkl')
    utils.save_pkl(encoding[1000:101000], f'{exp_dir_search_pool}adj_feat_matrix.pkl')
    utils.save_pkl(encoding[101000:], f'{exp_dir_pretrain}adj_feat_matrix.pkl')


if __name__ == '__main__':

    # args = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--sample_num", type=int, default=201000, help="random seed")
    parser.add_argument('--device_name', type=str, default='grid_16q', help='')
    parser.add_argument('--task', type=str, default='BeH_8', help='')
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

    # sample what circuit?
    # sample_pretrain_circuits(device_name, run_id, num_qubit, num_layer, num_sample=int(100))
    # sample_finetune_circuits(device_name, run_id, num_qubit, num_layer, num_sample=100)
    # sample_more_finetune_circuits(device_name, run_id, num_qubit, num_layer, num_sample=50)
    sample_search_circuits(device_name, task, run_id, num_qubit, num_layer, num_sample=args.sample_num)

    end_time = time.time()
    duration = end_time - start_time
    print(f"run time: {int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes")






