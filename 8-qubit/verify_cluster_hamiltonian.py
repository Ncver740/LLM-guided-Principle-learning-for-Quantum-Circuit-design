import tensorflow as tf
import numpy as np
import tensorcircuit as tc

tc.set_backend("tensorflow")
tc.set_dtype("complex128")


def build_cluster_hamiltonian(n_qubit, sparse=False):
    """
    构建 1D 横向场团簇模型 (Transverse-Field Cluster Model) 的哈密顿量。
    H = - sum_{j=1}^{n-2} Z_{j-1} X_j Z_{j+1} - X_0 Z_1 - Z_{n-2} X_{n-1}
    """
    pauli_terms = []
    coeffs = []

    # 主体部分: - sum_{j=1}^{n-2} Z_{j-1} X_j Z_{j+1}
    # 在 Python 索引中, j 对应 j, j-1 对应 j-1, j+1 对应 j+1
    for j in range(1, n_qubit - 1):
        term = [0] * n_qubit  # 0: I, 1: X, 2: Y, 3: Z
        term[j - 1] = 3  # Z_{j-1}
        term[j] = 1  # X_j
        term[j + 1] = 3  # Z_{j+1}
        pauli_terms.append(term)
        coeffs.append(-1.0)

    # 边界项 1: -X_0 Z_1
    boundary_term_1 = [0] * n_qubit
    boundary_term_1[0] = 1  # X_0
    boundary_term_1[1] = 3  # Z_1
    pauli_terms.append(boundary_term_1)
    coeffs.append(-1.0)

    # 边界项 2: -Z_{n-1} X_n (在 Python 索引中是 Z_{n-2} X_{n-1})
    boundary_term_2 = [0] * n_qubit
    boundary_term_2[n_qubit - 2] = 3  # Z_{n-2}
    boundary_term_2[n_qubit - 1] = 1  # X_{n-1}
    pauli_terms.append(boundary_term_2)
    coeffs.append(-1.0)

    pauli_terms_tf = tf.constant(pauli_terms, dtype=tf.int32)
    coeffs_tf = tf.constant(coeffs, dtype=tf.complex128)

    if sparse:
        return tc.quantum.PauliStringSum2COO(pauli_terms_tf, coeffs_tf)
    else:
        return tc.quantum.PauliStringSum2Dense(pauli_terms_tf, coeffs_tf)


# --- 验证函数 ---

def check_hamiltonian_properties(H, n_qubit):
    """
    方法一：检查哈密顿量的基本数学属性。
    """
    print("\n--- 方法一：正在检查哈密顿量的数学属性 ---")

    # 检查维度
    expected_dim = 2 ** n_qubit
    actual_dim = H.shape[0]
    print(f"1. 维度检查: 期望维度=({expected_dim}, {expected_dim}), 实际维度=({actual_dim}, {actual_dim})")
    assert expected_dim == actual_dim, "维度检查失败！"
    print("   ✅ 维度正确。")

    # 检查是否为实数矩阵
    is_real = np.all(np.isreal(H.numpy()))
    print(f"2. 实数检查: 矩阵是否为实数? {is_real}")
    assert is_real, "实数检查失败！"
    print("   ✅ 矩阵为实数。")

    # 检查对称性 (对于实数矩阵，对称等价于厄米)
    is_symmetric = np.allclose(H.numpy(), H.numpy().T)
    print(f"3. 对称性检查: 矩阵是否对称? {is_symmetric}")
    assert is_symmetric, "对称性检查失败！"
    print("   ✅ 矩阵是对称的。")

    # 检查迹
    trace = tf.linalg.trace(H)
    print(f"4. 迹检查: Tr(H) = {trace.numpy().real:.4f} (理论值为 0)")
    assert np.isclose(trace.numpy().real, 0), "迹检查失败！"
    print("   ✅ 迹为 0。")


def check_action_on_basis_state(H, n_qubit):
    """
    方法二：在一个简单的基矢上检查哈密顿量的作用。
    我们以 n=4 为例，计算 H |0000>。
    H = -X0Z1 - Z0X1Z2 - Z1X2Z3 - Z2X3
    -X0Z1 |0000> = -|1000>
    -Z0X1Z2 |0000> = -|0100>
    -Z1X2Z3 |0000> = -|0010>
    -Z2X3 |0000> = -|0001>
    所以, H|0000> = -|1> - |2> - |4> - |8> (十进制表示)
    """
    if n_qubit != 4:
        print("\n--- 方法二：跳过。此检查仅为 n_qubit=4 编写。 ---")
        return

    print(f"\n--- 方法二：正在检查 H 在 |0000> 上的作用 (n_qubit={n_qubit}) ---")

    # |0000> 态矢量
    state_vector = tf.constant([1.0] + [0.0] * (2 ** n_qubit - 1), dtype=tf.complex128)
    state_vector = tf.reshape(state_vector, (-1, 1))

    # 计算 H |psi>
    result_vector = tf.matmul(H, state_vector).numpy().flatten()

    # 找到非零元素的位置和值
    non_zero_indices = np.where(np.abs(result_vector) > 1e-9)[0]
    non_zero_values = result_vector[non_zero_indices]

    print("计算结果 H|0000> 的非零项:")
    for idx, val in zip(non_zero_indices, non_zero_values):
        print(f"  位置 {idx} (二进制: {idx:04b}), 值为: {val.real:.1f}")

    # 理论上的非零位置
    expected_indices = {1, 2, 4, 8}  # |0001>, |0010>, |0100>, |1000>

    assert set(non_zero_indices) == expected_indices, "非零元素位置不正确！"
    assert np.allclose(non_zero_values, -1.0), "非零元素的值不正确！"
    print("   ✅ H|0000> 的计算结果与手动推导一致。")


def check_ground_state_energy(H, n_qubit):
    """
    方法三：检查基态能量，这是最可靠的物理验证。
    """
    print(f"\n--- 方法三：正在检查基态能量 (n_qubit={n_qubit}) ---")

    # 理论基态能量
    theoretical_energy = -float(n_qubit)

    # 数值计算基态能量 (最小本征值)
    # tf.linalg.eigvalsh 专门用于计算厄米矩阵的本征值，速度快且稳定
    eigenvalues = tf.linalg.eigvalsh(H).numpy()
    numerical_energy = np.min(eigenvalues)

    print(f"理论基态能量 E₀ = {theoretical_energy}")
    print(f"数值计算的最小本征值 = {numerical_energy:.8f}")

    # 比较两者是否接近
    is_correct = np.isclose(theoretical_energy, numerical_energy)
    assert is_correct, "基态能量检查失败！"
    print("   ✅ 数值计算的基态能量与理论值匹配。")


if __name__ == "__main__":
    # --- 执行验证 ---
    # 您可以修改这个 n_qubit 来测试不同大小的系统
    N_QUBITS = 4

    print(f"****** 开始验证 {N_QUBITS}-qubit 的团簇模型哈密顿量 ******")

    # 1. 构建哈密顿量
    hamiltonian = build_cluster_hamiltonian(N_QUBITS)

    # 2. 执行所有检查
    check_hamiltonian_properties(hamiltonian, N_QUBITS)
    check_action_on_basis_state(hamiltonian, N_QUBITS)  # 这个是为 N=4 写的
    check_ground_state_energy(hamiltonian, N_QUBITS)

    print("\n****** 所有检查均已通过！哈密顿量构建正确。 ******")

    # 您也可以尝试更大的系统，但方法二会跳过
    print("\n\n****** 尝试一个更大的系统 (n=8) ******")
    N_QUBITS_LARGE = 8
    hamiltonian_large = build_cluster_hamiltonian(N_QUBITS_LARGE)
    check_hamiltonian_properties(hamiltonian_large, N_QUBITS_LARGE)
    check_action_on_basis_state(hamiltonian_large, N_QUBITS_LARGE)  # 会跳过
    check_ground_state_energy(hamiltonian_large, N_QUBITS_LARGE)