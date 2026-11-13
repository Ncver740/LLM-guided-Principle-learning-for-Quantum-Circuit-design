# --- START OF FILE circuit_sampler_all2all.py ---
""""
全连接-layerwise 方式采样量子线路
"""
import numpy as np
import random
import optimize_circuit
import config
import networkx as nx


class CircuitSamplerAll2All:
    def __init__(self, device_name, num_layer, num_qubit):
        '''
        专门针对全连接 (All-to-All) 拓扑的采样器
        :param device_name: e.g., 'full_10q'
        '''
        self.device = config.device[device_name]
        self.n_qubit = num_qubit
        self.num_layer = num_layer

        # 验证配置中的比特数是否一致
        if self.n_qubit != self.device['n_qubit']:
            print(f"Warning: Config qubit count ({self.device['n_qubit']}) != Target qubit count ({self.n_qubit})")

    def generate_qubit_mappings(self):
        """
        全连接拓扑下，物理比特和逻辑比特是等价的。
        不需要寻找连通子图，直接返回 Identity 映射。
        """
        # 逻辑比特 i 映射到 物理比特 i
        mapping = {i: i for i in range(self.n_qubit)}

        # 逻辑连接就是全连接
        logical_connections = self.device['connectivity']

        return mapping, logical_connections

    def generateCircuit(self):
        '''
        Layer-wise pipeline for All-to-All topology.
        相比原版，这里去掉了 get_maximal_valid_matchings，改为实时随机配对。
        '''

        # 全连接映射十分简单
        qubit_mapping, edges_mapping = self.generate_qubit_mappings()

        # 生成门类型的概率分布
        logit_gate_type = np.random.normal(size=len(self.device['gate_set']))
        prob_gate_type = np.exp(logit_gate_type) / sum(np.exp(logit_gate_type))

        # Layer-wise generation
        cir = []

        # 预先生成 qubit 索引列表
        qubit_indices = np.arange(self.n_qubit)

        for _ in range(self.num_layer):
            gate_type = np.random.choice(self.device['gate_set'], p=prob_gate_type)

            # 如果是双比特门 (CZ, CX, CNOT, etc.)
            if gate_type in ['cz', 'cx', 'cnot', 'rxx']:
                # === 核心修改：实时随机洗牌实现最大匹配 ===
                # 1. 打乱量子比特顺序
                shuffled_qubits = np.random.permutation(qubit_indices)

                # 2. 两两配对 (0,1), (2,3), ...
                # 如果是奇数个比特，最后一个会被丢弃，符合物理限制
                for i in range(0, self.n_qubit - 1, 2):
                    q0 = shuffled_qubits[i]
                    q1 = shuffled_qubits[i + 1]

                    # 确保索引小的在前 (或者针对 CX 随机反转)
                    if gate_type == 'cx' and random.choice([0, 1]) == 1:
                        q0, q1 = q1, q0

                    # 在全连接中，任意 q0, q1 都是相连的，直接添加
                    cir.append([gate_type, q0, q1])

            # 如果是单比特门
            else:
                for i in range(self.n_qubit):
                    cir.append([gate_type, i])

        return cir, qubit_mapping, edges_mapping

    def check_connectivity(self, cir, phy_to_log):
        '''
        验证线路是否符合全连接拓扑。
        在全连接配置下，只要比特索引不越界，理论上总是 True。
        为了严谨，我们依然查表。
        '''
        log_to_phy = {v: k for k, v in phy_to_log.items()}
        connectivity_set = set(tuple(sorted(edge)) for edge in self.device['connectivity'])

        for gate in cir:
            if len(gate) == 3:  # 2-qubit gate
                a = log_to_phy[gate[1]]
                b = log_to_phy[gate[2]]

                # 检查自身连接 (不合法)
                if a == b:
                    print(f'Error: Gate connected to self: {gate}')
                    return False

                edge = tuple(sorted([a, b]))
                if edge not in connectivity_set:
                    print(f'Error: Edge {edge} not in connectivity config.')
                    return False
        return True

    def sample(self, num_sample, result=None, qubit_mapping_lst=None, edges_mapping_list=None):
        '''
        采样主循环
        '''
        if result is None:
            result = []
            qubit_mapping_lst = []
            edges_mapping_list = []

        sample_count = 0
        while sample_count < num_sample:
            # 调用全连接专用的生成器
            cir, qubit_mapping, edges_mapping = self.generateCircuit()

            # 使用原有的 optimize 模块 (假设它能处理全连接生成的列表)，使用Qiskit的Transpile优化线路，level=1
            opt_cir = optimize_circuit.optimize(cir, self.device['gate_set'], num_qubit=self.n_qubit)

            if opt_cir[1] == 0:  # depth is zero
                continue

            # 连通性检查
            if not self.check_connectivity(opt_cir[0], qubit_mapping):
                print("Connectivity check failed! Check your config.")
                exit(123)

            if opt_cir not in result:
                result.append(opt_cir)
                edges_mapping_list.append(edges_mapping)
                qubit_mapping_lst.append(qubit_mapping)
                sample_count += 1
                if sample_count % 1000 == 0:
                    print(f"Generated {sample_count} samples...")

        return result, qubit_mapping_lst, edges_mapping_list