import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import utils
import argparse
import config
tc.set_dtype("complex128")
tc.set_backend("tensorflow")

class VqeTrainerNew:
    def __init__(self, n_cir_parallel, n_runs, max_iteration, n_qubit, hamiltonian, noise_param=None):
        self.K = tc.set_backend("tensorflow")
        self.n_qubit = n_qubit
        self.max_iteration = max_iteration
        self.n_cir_parallel = n_cir_parallel
        self.n_runs = n_runs

        self.h = hamiltonian
        self.give_up_rest = False
        self.solution = None

        """ Noise-related parameter, don't care if noise is False. """
        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
        if self.noise:
            self.two_qubit_channel_depolarizing_p = noise_param['two_qubit_channel_depolarizing_p']
            self.single_qubit_channel_depolarizing_p = noise_param['single_qubit_channel_depolarizing_p']
            self.bit_flip_p = noise_param['bit_flip_p']
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p/15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p/3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def compute_energy(self, param, structure):
        """
        :param param: Circuit Parameters
        :param structure: Circuit
        :return:
        """
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate[0] == "cx":
                    c.cx(gate[1], gate[2])
                    c.general_kraus(self.two_qubit_dep_channel, gate[1], gate[2])
                elif gate[0] == "cz":
                    c.cz(gate[1], gate[2])
                    c.general_kraus(self.two_qubit_dep_channel, gate[1], gate[2])
                elif gate[0] == "ry":
                    c.ry(gate[1], theta=param[param_index])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    param_index += 1
                elif gate[0] == "rz":
                    c.rz(gate[1], theta=param[param_index])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    param_index += 1
                elif gate[0] == "rx":
                    c.rx(gate[1], theta=param[param_index])
                    c.general_kraus(self.single_qubit_dep_channel, gate[1])
                    param_index += 1
                else:
                    print("invalid gate!")
                    exit(0)
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            """Calculate energy"""
            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)

        else:
            c = tc.Circuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate[0] == "cx":
                    c.cx(gate[1], gate[2])
                elif gate[0] == "cz":
                    c.cz(gate[1], gate[2])
                elif gate[0] == "ry":
                    c.ry(gate[1], theta=param[param_index])
                    param_index += 1
                elif gate[0] == "rz":
                    c.rz(gate[1], theta=param[param_index])
                    param_index += 1
                elif gate[0] == "rx":
                    c.rx(gate[1], theta=param[param_index])
                    param_index += 1
                else:
                    print("invalid gate!")
                    exit(0)
            e = tc.templates.measurements.operator_expectation(c, self.h)
        return e

    def get_param_num(self, cir):
        param_num = 0
        for i in range(len(cir)):
            if cir[i][0] == 'rx':
                param_num += 1
            if cir[i][0] == 'ry':
                param_num += 1
            if cir[i][0] == 'rz':
                param_num += 1
        return param_num

    def train_circuit(self, circuit_and_seed):
        single_circuit = circuit_and_seed[0]
        seed = circuit_and_seed[1]
        np.random.seed(seed)
        tf.random.set_seed(seed)

        param_num = self.get_param_num(single_circuit[0])
        trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0))
        L = single_circuit[1]
        par = np.random.normal(loc=0, scale=1 / (8 * (L + 2)), size=param_num)
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        param_initial = param.numpy()
        e_last = 1000
        energy_epoch = []
        opt = tf.keras.optimizers.Adam(0.05)
        if param_num > 0:
            for i in range(self.max_iteration):
                e, grad = trainer(param, single_circuit[0])
                energy_epoch.append(e.numpy())
                opt.apply_gradients([(grad, param)])
                if i % 100 == 0:
                    distance = abs(e_last - e.numpy())
                    if distance < 0.0001:
                        # print(distance.max())
                        break
                    else:
                        e_last = e.numpy()
        else:
            e, grad = trainer(param, single_circuit[0])
            energy_epoch = [e.numpy() for _ in range(self.max_iteration)]
        return e.numpy(), param.numpy(), energy_epoch

    # def draw(self, loss_list, exp_dir, circuit_id, best_index):
    #     '''
    #     :param loss_list: list, the train loss of a single training
    #     :param exp_dir: the directory to save data, e.g., 'result/run_1/'
    #     :param circuit_id: which circuit
    #     :return:
    #     '''
    #     epochs = range(1, len(loss_list) + 1)
    #     plt.figure(figsize=(10, 6))
    #     # 绘制训练损失
    #     plt.plot(epochs, loss_list, label='Training Loss', marker='o', markersize=1, color='blue')
    #     plt.title('Training Loss Over Epochs')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid()
    #
    #     # 调整布局并显示图形
    #     plt.tight_layout()
    #     save_path_img = f'{exp_dir}/circuit_train_curve/'
    #     if not os.path.exists(save_path_img):
    #         os.makedirs(save_path_img)
    #     # plt.show()
    #     plt.savefig(save_path_img + f'circuit_{circuit_id}_{best_index}.png')
    #     # plt.savefig(save_path_img + f'circuit_{circuit_id}_{best_index}.pdf',format='pdf')
    #     plt.savefig(save_path_img + f'circuit_{circuit_id}_{best_index}.svg', format='svg')
    #     plt.close()

    def draw(self, loss_list, exp_dir, circuit_id, best_index):
        """
        绘制单条训练损失曲线（npj Quantum Information 风格）
        - 无标题、无网格、白底
        - 线条平滑、清晰
        - 字体、线条粗细、比例均符合顶刊标准
        - 同时保存为 PNG 和 SVG
        """
        epochs = range(1, len(loss_list) + 1)

        # === 设置 npj Quantum Information 风格 ===
        plt.rcParams.update({
            "font.family": "serif",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "lines.linewidth": 2.5,
        })

        plt.figure(figsize=(8, 6))

        # === 绘制干净的loss曲线 ===
        plt.plot(
            epochs,
            loss_list,
            color="#1f77b4",  # 柔和蓝色（npj常用）
            linewidth=2.5,
            linestyle="-",
            marker=None,
        )

        # === 坐标轴标签 ===
        plt.xlabel("Epoch", fontsize=24)
        plt.ylabel("Loss", fontsize=24)

        # === 去掉标题、网格、图例 ===
        plt.title("")
        plt.grid(False)
        plt.legend().set_visible(False)

        # === 优化边界和布局 ===
        plt.tight_layout()
        plt.box(True)

        # === 保存路径 ===
        save_path_img = os.path.join(exp_dir, "circuit_train_curve")
        os.makedirs(save_path_img, exist_ok=True)

        # === 保存为高质量图像 ===
        png_path = os.path.join(save_path_img, f"circuit_{circuit_id}_{best_index}.png")
        svg_path = os.path.join(save_path_img, f"circuit_{circuit_id}_{best_index}.svg")

        plt.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")

        plt.close()

        print(f"✅ Saved loss curve to:\n- {png_path}\n- {svg_path}")

    def batch_train_parallel(self, device_name, task_name, run_id):
        start_time = time.time()
        # load finetune circuits
        exp_dir = f'cir_sample/{device_name}_{task_name}/training/run_{run_id}/'
        print(f"数据集路径：{exp_dir}")
        samples = utils.load_pkl(f'{exp_dir}samples.pkl')[:1000]

        # start training
        work_queue = []
        for i in range(0, len(samples)):
            print(f'circuit id: {i}')
            # load finetune circuits
            work_queue.extend([[samples[i], j] for j in range(0, self.n_runs)])

        pool = Pool(processes=self.n_cir_parallel)
        result = pool.map(self.train_circuit, work_queue)
        pool.close()
        pool.join()

        energy, param, energy_epoch = [], [], []
        for part in result:
            energy.append(part[0])
            param.append(part[1])
            energy_epoch.append(part[2])

        energy_f, param_f, energy_epoch_f = [], [], []
        for i in range(0, len(samples)):
            index0 = i * self.n_runs
            index1 = index0 + self.n_runs
            best_index = np.argmin(energy[index0:index1])
            best_index = best_index + index0
            energy_f.append(energy[best_index])
            param_f.append(param[best_index])
            energy_epoch_f.append(energy_epoch[best_index])
            self.draw(energy_epoch[best_index], exp_dir, i, best_index)

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        utils.save_pkl(energy_f, f'{exp_dir}energy.pkl')
        utils.save_pkl(param_f, f'{exp_dir}param.pkl')
        utils.save_pkl(energy_epoch_f, f'{exp_dir}energy_epoch.pkl')
        end_time = time.time()
        duration = end_time - start_time
        print(f"run time: {int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes")
        utils.save_pkl(duration, exp_dir + 'duration.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--noise", type=bool, default=False, help="是否考虑噪声")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="双qubit去极化噪声大小")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="单qubit去极化噪声大小")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="比特翻转噪声大小")
    args = parser.parse_args()

    device_name = 'full_4q'    # see config.py
    task_name = 'TFCluster_4'
    qubit = 4

    noise_param = None
    if args.noise:
        print("使用噪声训练 4q TFCluster 哈密顿量")
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}
    else:
        print("使用无噪声训练 4q TFCluster 哈密顿量")

    def convert_pauli_strings_to_numerical(pauli_strings):
        """
        辅助函数：将字符串表示的泡利算符列表转换为 TensorCircuit 的数值格式。
        例如: ['IX', 'ZI'] -> [[0, 1], [3, 0]]
        """
        pauli_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        numerical_terms = []
        for p_str in pauli_strings:
            numerical_terms.append([pauli_map[char.upper()] for char in p_str])
        return numerical_terms


    n_qubit = qubit # 比特数
    print(f"任务比特：{n_qubit}")
    # 我们将创建一个列表，其中每个元素都包含计算和打印所需的所有信息
    hamiltonian_terms = []

    # 添加边界项 1: -X_0 Z_1
    term_list_b1 = ['I'] * n_qubit
    term_list_b1[0] = 'X'
    term_list_b1[1] = 'Z'
    hamiltonian_terms.append({
        "string": "".join(term_list_b1),
        "coeff": -1.0,
        "type": "边界项",
        "indices": (0, 1)
    })

    # 构建主体部分: - sum_{j=1}^{n-2} Z_{j-1} X_j Z_{j+1}
    for j in range(1, n_qubit - 1):
        term_list = ['I'] * n_qubit
        term_list[j - 1] = 'Z'
        term_list[j] = 'X'
        term_list[j + 1] = 'Z'
        hamiltonian_terms.append({
            "string": "".join(term_list),
            "coeff": -1.0,
            "type": "主体项",
            "indices": (j - 1, j, j + 1)
        })

    # 添加边界项 2: -Z_{n-2} X_{n-1}
    term_list_b2 = ['I'] * n_qubit
    term_list_b2[n_qubit - 2] = 'Z'
    term_list_b2[n_qubit - 1] = 'X'
    hamiltonian_terms.append({
        "string": "".join(term_list_b2),
        "coeff": -1.0,
        "type": "边界项",
        "indices": (n_qubit - 2, n_qubit - 1)
    })

    # --- 打印部分：按位置排序后输出 ---
    # 使用 lambda 函数，根据每个项作用的第一个量子比特的索引进行排序
    sorted_terms_for_printing = sorted(hamiltonian_terms, key=lambda term: term["indices"][0])

    print("使用字符串定义的泡利项 (按物理位置排序):")
    for term in sorted_terms_for_printing:
        p_str = term["string"]
        term_type = term["type"]
        indices_str = ", ".join(map(str, term["indices"]))
        print(f"- {p_str}  ({term_type}，作用于 {indices_str})")

    # --- 计算部分：使用原始（或任意顺序的）列表 ---
    # 从 hamiltonian_terms 列表中提取计算所需的 pauli 字符串和系数
    pauli_strings_str = [term["string"] for term in hamiltonian_terms]
    coeffs = [term["coeff"] for term in hamiltonian_terms]

    # 1. 将字符串列表转换为数值列表
    pauli_terms_numerical = convert_pauli_strings_to_numerical(pauli_strings_str)

    # 2. 将数值列表转换为 TensorFlow 张量
    pauli_terms_tf = tf.constant(pauli_terms_numerical, dtype=tf.int32)
    coeffs_tf = tf.constant(coeffs, dtype=tf.complex128)

    # 3. 使用 TensorCircuit 函数构建哈密顿量矩阵
    hamiltonian_cluster_model = tc.quantum.PauliStringSum2Dense(pauli_terms_tf, coeffs_tf)

    print("\n成功构建横向场团簇模型的哈密顿量。")
    print("哈密顿量矩阵 (稠密形式) 的形状:", hamiltonian_cluster_model.shape)


    hamiltonian = hamiltonian_cluster_model
    trainer = VqeTrainerNew(n_cir_parallel=20, n_runs=10, max_iteration=2000, n_qubit=qubit, hamiltonian=hamiltonian, noise_param=noise_param)

    run_id = args.seed
    trainer.batch_train_parallel(device_name, task_name, run_id)

