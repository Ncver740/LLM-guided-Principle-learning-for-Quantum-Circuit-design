import config
from circuit_sampler_all2all import CircuitSamplerAll2All
import numpy as np


def verify_all_to_all_sampler():
    # 1. Setup
    device_name = 'full_4q'  # 确保你在 config.py 里加了这个
    n_qubit = 4
    n_layer = 10
    sampler = CircuitSamplerAll2All(device_name, n_layer, n_qubit)

    print(f"Starting verification for {device_name}...")

    # 2. Generate one raw circuit (without optimization to see layer structure)
    raw_cir, _, _ = sampler.generateCircuit()

    # 3. Check Layer-wise Logic manually
    # 我们知道 generateCircuit 是按层生成的，但输出是一个扁平列表。
    # 我们可以通过重现 generateCircuit 的随机种子或逻辑来检查，
    # 但更简单的方法是检查连续的双比特门块。

    print(f"Total gates generated: {len(raw_cir)}")

    current_layer_qubits = set()
    current_gate_type = raw_cir[0][0]

    layer_count = 0
    conflict_found = False

    # 简单的启发式检查：
    # 如果门类型改变，或者当前门的比特已经被同类型的门使用过，说明换层了
    # 注意：这只是针对未优化（raw）线路的粗略检查

    print("\n--- Checking for Qubit Conflicts within inferred layers ---")
    for gate in raw_cir:
        g_type = gate[0]

        # 这是一个新层吗？
        # 简化假设：如果类型变了，或者是双比特门且比特冲突了，或者是单比特门且比特冲突了
        qubits_in_gate = gate[1:]

        is_conflict = any(q in current_layer_qubits for q in qubits_in_gate)

        if g_type != current_gate_type or is_conflict:
            # 换层结算
            layer_count += 1
            current_layer_qubits = set()
            current_gate_type = g_type

        # 记录当前层的比特
        for q in qubits_in_gate:
            current_layer_qubits.add(q)

        # 再次检查当前门内部是否有冲突（比如 cz 1 1）
        if len(gate) == 3 and gate[1] == gate[2]:
            print(f"CRITICAL ERROR: Self-loop detected in gate {gate}")
            conflict_found = True

    print(f"Inferred approximately {layer_count} layers based on conflict boundaries.")

    if not conflict_found:
        print("SUCCESS: No structural conflicts found.")

    # 4. Check Distribution (Randomness)
    print("\n--- Checking Edge Distribution (100 samples) ---")
    samples, _, _ = sampler.sample(100)
    edge_counts = {}

    for sample in samples:
        circuit = sample[0]
        for gate in circuit:
            if len(gate) == 3:
                # 排序使得 (1,2) 和 (2,1) 视为同一条边
                edge = tuple(sorted((gate[1], gate[2])))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

    unique_edges = len(edge_counts)
    total_possible_edges = n_qubit * (n_qubit - 1) // 2
    print(f"Unique edges covered: {unique_edges} / {total_possible_edges}")

    if unique_edges > total_possible_edges * 0.8:
        print("SUCCESS: Good coverage of All-to-All topology.")
    else:
        print("WARNING: Coverage might be low, check iteration count or layers.")


if __name__ == "__main__":
    verify_all_to_all_sampler()