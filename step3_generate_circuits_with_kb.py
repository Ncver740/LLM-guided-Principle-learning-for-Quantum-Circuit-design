import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from enum import Enum

# Add current directory to path
os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
sys.path.insert(0, os.getcwd())

import json
import openai
import numpy as np
import pickle
import re
import random
import ast
import time
from datetime import datetime

# Import VQE training components
import tensorcircuit as tc
import tensorflow as tf
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# Set tensorcircuit backend
tc.set_dtype("complex128")
tc.set_backend("tensorflow")


# Enums and Configuration Classes

class ModelType(Enum):
    """Supported quantum model types"""
    HEISENBERG = "heisenberg"
    TFIM = "tfim"
    TFCM = "tfcm"


@dataclass
class Hyperparameters:
    """
    All configurable hyperparameters for the experiment.
    Adjust these to run top5, top10, top50 experiments.
    """
    # Circuit generation and evaluation
    top_k: int = 5  # Can be changed to 10, 50 for different experiments
    num_qubits: int = 8
    max_iteration: int = 2000
    n_runs: int = 10  # Training runs per circuit (consistent with vqe_task_paralell)

    # Coupling topology diversity
    min_coupling_topologies: int = 3

    # Energy thresholds
    energy_improvement_threshold: float = 0.2
    energy_convergence_threshold: float = 1e-4  # VQE convergence threshold (consistent with vqe_task_paralell: 0.0001)

    # Noise parameters
    two_qubit_depolarizing_p: float = 0.01
    single_qubit_depolarizing_p: float = 0.001
    bit_flip_p: float = 0.01

    # Prompt settings
    max_prompt_chars: int = 50000
    max_knowledge_chars: int = 50000
    max_principles_chars: int = 3500
    max_circuit_repr_chars: int = 2000
    max_generation_attempts: int = 3


# Global hyperparameters instance
HP = Hyperparameters()


# API Configuration
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL = "YOUR_MODEL_NAME"

# Data Paths Configuration

# Heisenberg Model Configuration
HEISENBERG_CONFIG = {
    "model_name": "Heisenberg",
    "principles_path": "YOUR_PATH",
    "output_dir": "YOUR_PATH",
}

# TFIM Model Configuration
TFIM_CONFIG = {
    "model_name": "TFIM",
    "principles_path": "YOUR_PATH",
    "output_dir": "YOUR_PATH",
}

# TFCM Model Configuration
TFCM_CONFIG = {
    "model_name": "TFCM",
    "principles_path": "YOUR_PATH",
    "output_dir": "YOUR_PATH",
}

# Default configuration (can be changed via command line)
CURRENT_CONFIG = TFCM_CONFIG


# Hamiltonian Configuration

def get_hamiltonian_config(model_type: ModelType) -> Dict:
    """
    Get Hamiltonian configuration for specified model type.

    Models differ in their coupling terms:

    1. Heisenberg Model:
       H = J Σ (σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1} + σ_z^i σ_z^{i+1})
       - Isotropic interaction in all three directions
       - Ground state energy (noisy): ~-4
       - Ground state energy (ideal): ~-8

    2. TFIM (Transverse Field Ising Model):
       H = -J Σ σ_z^i σ_z^{i+1} - h Σ σ_x^i
       - Z-direction interaction with X-direction transverse field
       - Quantum phase transition at h_c/J = 1
       - Ground state energy (noisy): ~-3.5
       - Ground state energy (ideal): ~-7

    3. TFCM (Transverse Field Cluster Model):
       H = -J Σ (σ_z^i σ_z^{i+1} + λ σ_x^i σ_x^{i+1}) - h Σ σ_x^i
       - Cluster interaction (both ZZ and XX terms)
       - Adjustable λ controls cluster XX coupling strength
       - Ground state energy (noisy): ~-3.8
       - Ground state energy (ideal): ~-7.5
    """
    configs = {
        ModelType.HEISENBERG: {
            "name": "Heisenberg",
            "description": "Isotropic Heisenberg Model",
            "hzz": 1.0,
            "hxx": 1.0,
            "hyy": 1.0,
            "hx": 0.0,
            "hy": 0.0,
            "hz": 1.0,
            "target_energy_noisy": -11.5451,
            "target_energy_ideal": -8.0,
            "circuit_depth_range": (15, 25),
            "parameterized_gates_range": (20, 35),
            "two_qubit_gates_range": (8, 15),
        },
        ModelType.TFIM: {
            "name": "TFIM",
            "description": "Transverse Field Ising Model",
            "hzz": 1.0,
            "hxx": 0.0,
            "hyy": 0.0,
            "hx": 1.0,
            "hy": 0.0,
            "hz": 0.0,
            "target_energy_noisy": -9.74519,
            "target_energy_ideal": -7.0,
            "circuit_depth_range": (12, 28),
            "parameterized_gates_range": (18, 35),
            "two_qubit_gates_range": (8, 16),
        },
        ModelType.TFCM: {
            "name": "TFCM",
            "description": "Transverse Field Cluster Model",
            # TFCM (Cluster Model) Hamiltonian: H = -Σ Z_{{i-1}} X_i Z_{{i+1}} - h Σ X_i
            # This is a three-site interaction model (cluster model), not conventional two-body
            # Since tensorcircuit's heisenberg_hamiltonian doesn't support ZXZ, using equivalent params
            # For 8-qubit system, dataset ground state energy is -6.37818
            "hzz": 0.0,
            "hxx": 1.0,
            "hyy": 0.0,
            "hx": 1.0,
            "hy": 0.0,
            "hz": 0.0,
            "target_energy_noisy": -6.37818,
            "target_energy_ideal": -7.0,
            "circuit_depth_range": (15, 35),
            "parameterized_gates_range": (20, 40),
            "two_qubit_gates_range": (10, 20),
        }
    }
    return configs.get(model_type, configs[ModelType.HEISENBERG])


# Model-Specific Prompt Templates (English)

def get_prompt_templates(model_type: ModelType) -> Dict:
    """
    Get English prompt templates for specified model type.
    Prompts are task-specific and optimized for each Hamiltonian.
    """
    templates = {
        ModelType.HEISENBERG: {
            "system_prompt": """You design 8-qubit Heisenberg VQE circuits.
Return ONLY a Python list.
Allowed gates: rx, ry, rz, cx, cz.
Each single-qubit gate is [gate_type, qubit].
Each two-qubit gate is [gate_type, qubit1, qubit2].""",

            "user_prompt_template": """Generate {num_circuits} different 8-qubit Heisenberg circuits.

Requirements:
- Use exactly {num_qubits} qubits
- Use 20-35 parameterized gates and 8-15 two-qubit gates
- Keep depth around 15-25
- Prefer efficient entanglement and diverse topologies
- Return ONLY a Python list with no explanation

Knowledge base:
{knowledge_base}

Design principles:
{principles}"""
        },

        ModelType.TFIM: {
            "system_prompt": """You design 8-qubit TFIM VQE circuits.
Return ONLY a Python list.
Allowed gates: rx, ry, rz, cx, cz.
Each single-qubit gate is [gate_type, qubit].
Each two-qubit gate is [gate_type, qubit1, qubit2].""",

            "user_prompt_template": """Generate {num_circuits} different 8-qubit TFIM circuits.

Requirements:
- Use exactly {num_qubits} qubits
- Use 18-35 parameterized gates and 8-16 two-qubit gates
- Keep depth around 12-28
- Include enough RX rotations and efficient entanglement
- Return ONLY a Python list with no explanation

Knowledge base:
{knowledge_base}

Design principles:
{principles}"""
        },

        ModelType.TFCM: {
            "system_prompt": """You design 8-qubit TFCM VQE circuits.
Return ONLY a Python list.
Allowed gates: rx, ry, rz, cx, cz.
Each single-qubit gate is [gate_type, qubit].
Each two-qubit gate is [gate_type, qubit1, qubit2].""",

            "user_prompt_template": """Generate {num_circuits} different 8-qubit TFCM circuits.

Requirements:
- Use exactly {num_qubits} qubits
- Use 20-40 parameterized gates and 10-20 two-qubit gates
- Keep depth around 15-35
- Include enough RX rotations and cluster-like entanglement patterns
- Return ONLY a Python list with no explanation

Knowledge base:
{knowledge_base}

Design principles:
{principles}"""
        }
    }
    return templates.get(model_type, templates[ModelType.HEISENBERG])


# Helper Functions

def load_pkl(file_path):
    """Load pkl file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pkl(data, file_path):
    """Save data to pkl file"""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def validate_file_exists(file_path: str, file_description: str) -> bool:
    """Validate that a required file exists"""
    if not os.path.exists(file_path):
        print("=" * 80)
        print(f"ERROR: {file_description} not found!")
        print(f"Expected path: {os.path.abspath(file_path)}")
        print("Please ensure the file exists before running this script.")
        print("=" * 80)
        sys.exit(1)
    return True


def load_knowledge_base(knowledge_path: str) -> str:
    """Load COMPLETE knowledge base (NO TRUNCATION)"""
    validate_file_exists(knowledge_path, "Knowledge base file")
    print(f"Loading COMPLETE knowledge base from: {knowledge_path}")
    with open(knowledge_path, 'r', encoding='utf-8') as f:
        knowledge = f.read()
    print(f"Loaded knowledge base ({len(knowledge)} characters) - FULL CONTENT")
    return knowledge


def load_design_principles(principles_path: str) -> str:
    """Load COMPLETE design principles (NO TRUNCATION)"""
    validate_file_exists(principles_path, "Design principles file")
    print(f"Loading COMPLETE design principles from: {principles_path}")
    with open(principles_path, 'r', encoding='utf-8') as f:
        principles = f.read()
    print(f"Loaded design principles ({len(principles)} characters) - FULL CONTENT")
    return principles


def compress_prompt_text(text: str, max_chars: int) -> str:
    """Compact prompt text while keeping the original order of information"""
    compact_text = re.sub(r'\n\s*\n+', '\n\n', text.strip())
    if len(compact_text) <= max_chars:
        return compact_text
    return compact_text[:max_chars].rsplit('\n', 1)[0].strip()


def check_circuit_validity(circuit: List, n_qubits: int) -> Tuple[bool, str]:
    """Check circuit validity before training"""
    if not isinstance(circuit, list) or len(circuit) == 0:
        return False, "Circuit must be a non-empty list"

    valid_gates = {'rx', 'ry', 'rz', 'cx', 'cz'}
    for gate in circuit:
        if not isinstance(gate, list) or len(gate) < 2:
            return False, f"Gate format error: {gate}"

        gate_type = gate[0]
        if gate_type not in valid_gates:
            return False, f"Invalid gate type: {gate_type}"

        if gate_type in {'rx', 'ry', 'rz'}:
            if len(gate) != 2:
                return False, f"Single-qubit gate format error: {gate}"
            qubit = gate[1]
            if not isinstance(qubit, int) or not 0 <= qubit < n_qubits:
                return False, f"Single-qubit gate qubit index out of range: {gate}"
        else:
            if len(gate) != 3:
                return False, f"Two-qubit gate format error: {gate}"
            q1, q2 = gate[1], gate[2]
            if not isinstance(q1, int) or not isinstance(q2, int):
                return False, f"Two-qubit gate qubit index type error: {gate}"
            if not (0 <= q1 < n_qubits and 0 <= q2 < n_qubits):
                return False, f"Two-qubit gate qubit index out of range: {gate}"
            if q1 == q2:
                return False, f"Two-qubit gate uses the same qubit twice: {gate}"

    return True, ""


def filter_valid_circuits(circuits: List, n_qubits: int) -> List[List]:
    """Keep only valid and trainable LLM circuits"""
    valid_circuits = []
    for idx, circuit in enumerate(circuits, 1):
        is_valid, reason = check_circuit_validity(circuit, n_qubits)
        if not is_valid:
            print(f"Filter circuit {idx}: {reason}")
            continue

        if sum(1 for gate in circuit if gate[0] in {'rx', 'ry', 'rz'}) == 0:
            print(f"Filter circuit {idx}: missing parameterized gates")
            continue

        valid_circuits.append(circuit)

    return valid_circuits


# Prompt Building Functions

def build_optimized_prompt(knowledge_base: str, principles: str, num_circuits: int,
                          num_qubits: int, model_type: ModelType) -> Tuple[str, str]:
    """Build optimized prompt for uniform circuit generation (model-specific)"""
    templates = get_prompt_templates(model_type)

    compact_knowledge = compress_prompt_text(knowledge_base, HP.max_knowledge_chars)
    compact_principles = compress_prompt_text(principles, HP.max_principles_chars)
    system_prompt = templates["system_prompt"]
    user_prompt = templates["user_prompt_template"].format(
        num_circuits=num_circuits,
        num_qubits=num_qubits,
        knowledge_base=compact_knowledge,
        principles=compact_principles
    )

    total_length = len(system_prompt) + len(user_prompt)
    if total_length > HP.max_prompt_chars:
        overflow = total_length - HP.max_prompt_chars
        compact_knowledge = compress_prompt_text(knowledge_base, max(1500, HP.max_knowledge_chars - overflow))
        user_prompt = templates["user_prompt_template"].format(
            num_circuits=num_circuits,
            num_qubits=num_qubits,
            knowledge_base=compact_knowledge,
            principles=compact_principles
        )

    return system_prompt, user_prompt


def build_circuit_by_circuit_prompt(knowledge_base: str, principles: str, num_circuits: int,
                                   num_qubits: int, model_type: ModelType) -> Tuple[str, str]:
    """Alternative prompt: Generate each circuit INDIVIDUALLY with specific optimization targets"""
    ham_config = get_hamiltonian_config(model_type)

    circuit_targets = [
        f"Linear nearest-neighbor coupling, balanced depth, {ham_config['name']} specific",
        f"Skip-one coupling pattern, deeper circuit, high expressibility",
        f"Pair coupling topology, shallow circuit, efficient entanglement",
        f"Ring-jump coupling, medium depth, noise-resilient design",
        f"Cross-coupling pattern, variable depth, optimized for training",
    ]

    circuit_strategy_lines = []
    for i, target in enumerate(circuit_targets[:num_circuits]):
        circuit_strategy_lines.append(f"- Circuit {i+1}: {target}")
    circuit_strategy_text = "\n".join(circuit_strategy_lines)

    circuit_spec_lines = []
    for i in range(min(num_circuits, len(circuit_targets))):
        circuit_spec_lines.append(f"**Circuit {i+1}**: {circuit_targets[i]}")
    circuit_spec_text = "\n".join(circuit_spec_lines)

    compact_knowledge = compress_prompt_text(knowledge_base, HP.max_knowledge_chars)
    compact_principles = compress_prompt_text(principles, HP.max_principles_chars)
    system_prompt = f"""You design {num_circuits} {ham_config['name']} VQE circuits.
Return ONLY a Python list of circuits.
Allowed gates: rx, ry, rz, cx, cz.
Use {ham_config['parameterized_gates_range'][0]}-{ham_config['parameterized_gates_range'][1]} parameterized gates,
{ham_config['two_qubit_gates_range'][0]}-{ham_config['two_qubit_gates_range'][1]} two-qubit gates,
and depth around {ham_config['circuit_depth_range'][0]}-{ham_config['circuit_depth_range'][1]}.
{circuit_strategy_text}"""

    user_prompt = f"""Generate exactly {num_circuits} different {ham_config['name']} circuits.

Knowledge base:
{compact_knowledge}

Design principles:
{compact_principles}

Circuit specifications:
{circuit_spec_text}

Return ONLY a Python list. Each circuit must use [gate_type, qubit] or [gate_type, qubit1, qubit2]."""

    total_length = len(system_prompt) + len(user_prompt)
    if total_length > HP.max_prompt_chars:
        overflow = total_length - HP.max_prompt_chars
        compact_knowledge = compress_prompt_text(knowledge_base, max(1500, HP.max_knowledge_chars - overflow))
        user_prompt = f"""Generate exactly {num_circuits} different {ham_config['name']} circuits.

Knowledge base:
{compact_knowledge}

Design principles:
{compact_principles}

Circuit specifications:
{circuit_spec_text}

Return ONLY a Python list. Each circuit must use [gate_type, qubit] or [gate_type, qubit1, qubit2]."""

    return system_prompt, user_prompt


# LLM API Functions

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call LLM API"""
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"API Base URL: {client.base_url}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"\nCalling LLM (prompt length: {len(system_prompt) + len(user_prompt)} chars)...")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            n=1
        )

        result = response.choices[0].message.content
        print("Successfully received response from LLM")
        return result

    except Exception as e:
        print(f"Error calling LLM API: {e}")
        raise


def parse_generated_circuits(response: str) -> List[List]:
    """Parse LLM returned circuits"""
    try:
        lines = response.strip().split('\n')
        circuit_text = None

        for i, line in enumerate(lines):
            if '[[' in line:
                start_idx = i
                bracket_count = line.count('[') - line.count(']')
                j = i + 1
                while j < len(lines) and bracket_count > 0:
                    bracket_count += lines[j].count('[') - lines[j].count(']')
                    j += 1
                if bracket_count == 0:
                    circuit_text = '\n'.join(lines[start_idx:j])
                    break

        if circuit_text is None:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                circuit_text = match.group()

        if circuit_text:
            circuits = ast.literal_eval(circuit_text)
            print(f"Successfully parsed {len(circuits)} circuits")
            return circuits
        else:
            print("Could not parse circuits from response")
            return []

    except Exception as e:
        print(f"Error parsing circuits: {e}")
        return []


def depth_count(cir, n_qubit):
    """Calculate circuit depth, consistent with vqe_task_paralell and test_xl.py."""
    res = [0] * n_qubit
    for gate in cir:
        gate_name = gate[0]
        if gate_name in ['cx', 'cz']:  # Two-qubit gate
            qubits = [gate[1], gate[2]]
            depth_q = [res[q] for q in qubits]
            max_depth = max(depth_q) + 1
            for q in qubits:
                res[q] = max_depth
        else:  # Single-qubit gate
            qubit = gate[1]
            res[qubit] += 1
    return np.max(res) if res else 0


def calculate_circuit_stats(circuit: List) -> Tuple[int, int, int]:
    """Calculate circuit statistics"""
    depth = 0
    num_gate = len(circuit)
    num_2q_gate = 0

    for gate in circuit:
        if len(gate) == 3:
            num_2q_gate += 1

    return depth, num_gate, num_2q_gate


# VQE Trainer Class

class VQETrainer:
    """VQE Trainer with multi-model support and noise simulation"""

    def __init__(self, n_qubit: int, model_type: ModelType = ModelType.HEISENBERG,
                 max_iteration: int = 2000, n_runs: int = 10,
                 noise: bool = True, two_qubit_depolarizing_p: float = 0.01,
                 single_qubit_depolarizing_p: float = 0.001, bit_flip_p: float = 0.01):

        self.n_qubit = n_qubit
        self.model_type = model_type
        self.max_iteration = max_iteration
        self.n_runs = n_runs
        self.noise = noise

        # Get Hamiltonian configuration
        ham_config = get_hamiltonian_config(model_type)

        self.K = tc.set_backend("tensorflow")

        # Create Hamiltonian based on model type
        if model_type == ModelType.TFCM:
            # TFCM uses custom transverse field cluster model Hamiltonian
            self.h = self._build_tfcm_hamiltonian(n_qubit)
            print(f"VQE Trainer initialized:")
            print(f"  - Model: {ham_config['name']} (Transverse Field Cluster Model)")
            print(f"  - Qubits: {n_qubit}")
            print(f"  - Hamiltonian shape: {self.h.shape}")
        else:
            # Heisenberg and TFIM use standard Hamiltonian
            self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=True)
            self.h = tc.quantum.heisenberg_hamiltonian(
                self.lattice,
                hzz=ham_config['hzz'],
                hxx=ham_config['hxx'],
                hyy=ham_config['hyy'],
                hx=ham_config['hx'],
                hy=ham_config['hy'],
                hz=ham_config['hz'],
                sparse=False
            )
            print(f"VQE Trainer initialized:")
            print(f"  - Model: {ham_config['name']} ({ham_config['description']})")
            print(f"  - Qubits: {n_qubit}")
            print(f"  - Target energy (noisy): ~{ham_config['target_energy_noisy']}")
            print(f"  - Target energy (ideal): ~{ham_config['target_energy_ideal']}")

        if self.noise:
            self.two_qubit_channel_depolarizing_p = two_qubit_depolarizing_p
            self.single_qubit_channel_depolarizing_p = single_qubit_depolarizing_p
            self.bit_flip_p = bit_flip_p

            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.two_qubit_channel_depolarizing_p / 15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)

            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.single_qubit_channel_depolarizing_p / 3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

            print(f"  Noise enabled: 2q_p={two_qubit_depolarizing_p}, 1q_p={single_qubit_depolarizing_p}")
        else:
            print("  Noise disabled")


    def _build_tfcm_hamiltonian(self, n_qubit):
        """Build the transverse-field cluster-model Hamiltonian, consistent with test_xl.py.
        H = -X_0 Z_1 - sum_{j=1}^{n-2} Z_{j-1} X_j Z_{j+1} - Z_{n-2} X_{n-1}

        For an 8-qubit system, this is a three-body interaction model (cluster model).
        """
        pauli_map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
        hamiltonian_terms = []

        # Boundary term 1: -X_0 Z_1
        term_list_b1 = ['I'] * n_qubit
        term_list_b1[0] = 'X'
        term_list_b1[1] = 'Z'
        hamiltonian_terms.append(("".join(term_list_b1), -1.0))

        # Bulk term: -sum_{j=1}^{n-2} Z_{j-1} X_j Z_{j+1}
        for j in range(1, n_qubit - 1):
            term_list = ['I'] * n_qubit
            term_list[j - 1] = 'Z'
            term_list[j] = 'X'
            term_list[j + 1] = 'Z'
            hamiltonian_terms.append(("".join(term_list), -1.0))

        # Boundary term 2: -Z_{n-2} X_{n-1}
        term_list_b2 = ['I'] * n_qubit
        term_list_b2[n_qubit - 2] = 'Z'
        term_list_b2[n_qubit - 1] = 'X'
        hamiltonian_terms.append(("".join(term_list_b2), -1.0))

        # Convert to a numeric representation
        pauli_strings_str = [term[0] for term in hamiltonian_terms]
        coeffs = [term[1] for term in hamiltonian_terms]

        pauli_terms_numerical = []
        for p_str in pauli_strings_str:
            pauli_terms_numerical.append([pauli_map[char.upper()] for char in p_str])

        pauli_terms_tf = tf.constant(pauli_terms_numerical, dtype=tf.int32)
        coeffs_tf = tf.constant(coeffs, dtype=tf.complex128)

        # Build the Hamiltonian matrix
        hamiltonian_matrix = tc.quantum.PauliStringSum2Dense(pauli_terms_tf, coeffs_tf)

        return hamiltonian_matrix

    def get_param_num(self, cir: List) -> int:
        param_num = 0
        for gate in cir:
            if gate[0] in ['rx', 'ry', 'rz']:
                param_num += 1
        return param_num

    def compute_energy(self, param, structure):
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            param_index = 0

            for gate in structure:
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

            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)

        else:
            c = tc.Circuit(self.n_qubit)
            param_index = 0

            for gate in structure:
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

            e = tc.templates.measurements.operator_expectation(c, self.h)

        return e

    def train_circuit(self, circuit: List, seed: int = 42):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        param_num = self.get_param_num(circuit)
        if param_num == 0:
            return float('inf'), None, []

        trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0))

        # Use depth_count for parameter initialization (consistent with vqe_task_paralell and test_xl.py)
        L = depth_count(circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1/(8*(L+2)), size=param_num)
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )

        opt = tf.keras.optimizers.Adam(0.05)

        e_last = 1000
        energy_epoch = []

        for i in range(self.max_iteration):
            e, grad = trainer(param, circuit)
            energy_epoch.append(e.numpy())
            opt.apply_gradients([(grad, param)])

            if i % 100 == 0:
                distance = abs(e_last - e.numpy())
                if distance < HP.energy_convergence_threshold:
                    break
                e_last = e.numpy()

        return e.numpy(), param.numpy(), energy_epoch

    def train_multiple_runs(self, circuit: List) -> Tuple[float, np.ndarray, List]:
        best_energy = float('inf')
        best_params = None
        best_energy_epoch = []

        for run_id in range(self.n_runs):
            seed = run_id  # Consistent with vqe_task_paralell and test_xl.py
            energy, params, energy_epoch = self.train_circuit(circuit, seed=seed)

            if energy < best_energy:
                best_energy = energy
                best_params = params
                best_energy_epoch = energy_epoch

            print(f"  Run {run_id+1}/{self.n_runs}: Energy = {energy:.6f}")

        return best_energy, best_params, best_energy_epoch


def format_circuit_output(circuit: List) -> List:
    """Format circuit output to samples.pkl format"""
    depth, num_gate, num_2q_gate = calculate_circuit_stats(circuit)

    depth = 0
    current_layer_ops = []
    for gate in circuit:
        if len(gate) == 3:
            if current_layer_ops:
                depth += 1
                current_layer_ops = []
            depth += 1
        else:
            current_layer_ops.append(gate)
    if current_layer_ops:
        depth += 1

    return [circuit, depth, num_gate, num_2q_gate]


def extract_coupling_topology(circuit: List) -> set:
    """Extract coupling topology from circuit"""
    couplings = set()
    for gate in circuit:
        if len(gate) == 3:
            q1, q2 = min(gate[1], gate[2]), max(gate[1], gate[2])
            couplings.add((q1, q2))
    return couplings


def create_default_circuits(num_circuits: int = 5, num_qubits: int = 8) -> List[List]:
    """Create default circuits with diverse coupling"""
    circuits = []

    topologies = [
        [(i, (i+1) % num_qubits) for i in range(num_qubits)],  # Linear
        [(i, (i+2) % num_qubits) for i in range(num_qubits)],  # Skip-1
        [(i, i + num_qubits // 2) for i in range(num_qubits // 2)],  # Pair
    ]

    for idx in range(num_circuits):
        circuit = []
        topology = topologies[idx % len(topologies)]

        for q in range(num_qubits):
            gate_type = ['rx', 'ry', 'rz'][q % 3]
            circuit.append([gate_type, q])

        for q1, q2 in topology:
            if q1 < num_qubits and q2 < num_qubits:
                circuit.append(['cz', q1, q2])

        circuits.append(circuit)

    return circuits


# Main Function

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Quantum Circuit Design Principle Learning v35')

    # Model type selection
    parser.add_argument("--model", type=str, default="heisenberg",
                       choices=["heisenberg", "tfim", "tfcm"],
                       help="Model type: heisenberg, tfim, or tfcm")

    # Hyperparameters
    parser.add_argument("--top_k", type=int, default=HP.top_k,
                       help=f"Number of circuits to generate (default: {HP.top_k})")
    parser.add_argument("--prompt_strategy", type=str, default="uniform",
                       choices=["uniform", "individual"],
                       help="Prompt strategy: 'uniform' or 'individual'")
    parser.add_argument("--noise", type=bool, default=True,
                       help="Enable noise simulation")
    parser.add_argument("--two_qubit_depolarizing_p", type=float,
                       default=HP.two_qubit_depolarizing_p,
                       help="Two-qubit depolarizing noise")
    parser.add_argument("--single_qubit_depolarizing_p", type=float,
                       default=HP.single_qubit_depolarizing_p,
                       help="Single-qubit depolarizing noise")
    parser.add_argument("--bit_flip_p", type=float,
                       default=HP.bit_flip_p,
                       help="Bit-flip noise")

    args = parser.parse_args()

    # Parse model type
    model_type_str = args.model.lower()
    if model_type_str == "heisenberg":
        model_type = ModelType.HEISENBERG
        config = HEISENBERG_CONFIG
    elif model_type_str == "tfim":
        model_type = ModelType.TFIM
        config = TFIM_CONFIG
    elif model_type_str == "tfcm":
        model_type = ModelType.TFCM
        config = TFCM_CONFIG
    else:
        model_type = ModelType.TFIM
        config = TFIM_CONFIG

    ham_config = get_hamiltonian_config(model_type)

    print("=" * 80)
    print("Quantum Circuit Design Principle Learning v35")
    print("Multi-Model Knowledge-Enhanced Circuit Generation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Model: {ham_config['name']} ({ham_config['description']})")
    print(f"  - TOP_K: {args.top_k}")
    print(f"  - Prompt Strategy: {args.prompt_strategy}")
    print(f"  - Noise: {args.noise}")
    print(f"  - Output Directory: {config['output_dir']}")
    print("=" * 80)

    # Update hyperparameters from command line arguments
    HP.top_k = args.top_k
    HP.two_qubit_depolarizing_p = args.two_qubit_depolarizing_p
    HP.single_qubit_depolarizing_p = args.single_qubit_depolarizing_p
    HP.bit_flip_p = args.bit_flip_p

    top_k = args.top_k
    prompt_strategy = args.prompt_strategy
    noise_enabled = args.noise

    # Step 1: Load files
    print("\n[Step 1] Loading files...")
    knowledge_base_path = "YOUR_PATH"
    if not os.path.exists(knowledge_base_path):
        knowledge_base_path = "YOUR_PATH"

    knowledge_base = load_knowledge_base(knowledge_base_path)
    principles = load_design_principles(config["principles_path"])
    print("✓ All required files loaded successfully")

    # Step 2: Generate circuits
    print(f"\n[Step 2] Generating circuits with '{prompt_strategy}' strategy...")

    circuits = []
    seen_circuits = set()
    sys_prompt = ""
    user_prompt = ""

    for attempt in range(HP.max_generation_attempts):
        remaining = top_k - len(circuits)
        if remaining <= 0:
            break

        print(f"LLM generation attempt {attempt + 1}/{HP.max_generation_attempts}, need {remaining} more circuits")
        if prompt_strategy == "uniform":
            sys_prompt, user_prompt = build_optimized_prompt(
                knowledge_base, principles, remaining, HP.num_qubits, model_type
            )
        else:
            sys_prompt, user_prompt = build_circuit_by_circuit_prompt(
                knowledge_base, principles, remaining, HP.num_qubits, model_type
            )

        llm_response = call_llm(sys_prompt, user_prompt)
        parsed_circuits = parse_generated_circuits(llm_response)
        valid_circuits = filter_valid_circuits(parsed_circuits, HP.num_qubits)

        for circuit in valid_circuits:
            circuit_key = tuple(tuple(gate) for gate in circuit)
            if circuit_key in seen_circuits:
                continue
            seen_circuits.add(circuit_key)
            circuits.append(circuit)
            if len(circuits) >= top_k:
                break

    if not circuits:
        raise RuntimeError("Failed to generate any valid and trainable LLM circuits. Training aborted.")

    circuits = circuits[:top_k]
    print(f"\nGenerated {len(circuits)} valid LLM circuits")

    # Step 3: Train circuits
    print("\n[Step 3] Training circuits with VQE...")
    trainer = VQETrainer(
        n_qubit=HP.num_qubits,
        model_type=model_type,
        max_iteration=HP.max_iteration,
        n_runs=HP.n_runs,
        noise=args.noise,
        two_qubit_depolarizing_p=args.two_qubit_depolarizing_p,
        single_qubit_depolarizing_p=args.single_qubit_depolarizing_p,
        bit_flip_p=args.bit_flip_p
    )

    results = []
    all_energies = []

    for i, circuit in enumerate(circuits):
        print(f"\n--- Training Circuit {i+1}/{len(circuits)} ---")
        best_energy, best_params, energy_epoch = trainer.train_multiple_runs(circuit)

        circuit_output = format_circuit_output(circuit)
        result = {
            'circuit': circuit_output,
            'circuit_list': circuit,
            'depth': circuit_output[1],
            'num_gate': circuit_output[2],
            'num_2q_gate': circuit_output[3],
            'converged_energy': best_energy,
            'best_params': best_params,
            'energy_epoch': energy_epoch,
            'model_type': model_type.value,
            'coupling_topology': extract_coupling_topology(circuit)
        }

        results.append(result)
        all_energies.append(best_energy)
        print(f"Circuit {i+1} converged energy: {best_energy:.6f}")

    # Step 4: Save results
    print("\n[Step 4] Saving results...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Sort results by energy
    sorted_results = sorted(results, key=lambda x: x['converged_energy'])

    # Save in samples.pkl format
    samples_output = [r['circuit'] for r in results]
    samples_path = os.path.join(output_dir, "generated_samples.pkl")
    save_pkl(samples_output, samples_path)

    # Save energies
    energy_path = os.path.join(output_dir, "generated_energy.pkl")
    save_pkl(all_energies, energy_path)

    # Save full results
    full_results_path = os.path.join(output_dir, "full_results.pkl")
    save_pkl(results, full_results_path)

    # Save metadata
    metadata = {
        'model_type': model_type.value,
        'model_name': ham_config['name'],
        'top_k': top_k,
        'prompt_strategy': prompt_strategy,
        'timestamp': timestamp,
        'prompt_length': len(sys_prompt) + len(user_prompt),
        'knowledge_base_length': len(knowledge_base),
        'principles_length': len(principles),
        'noise_params': {
            'noise': args.noise,
            'two_qubit_depolarizing_p': args.two_qubit_depolarizing_p,
            'single_qubit_depolarizing_p': args.single_qubit_depolarizing_p,
            'bit_flip_p': args.bit_flip_p
        }
    }
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    save_pkl(metadata, metadata_path)

    # Step 5: Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  - Model: {ham_config['name']}")
    print(f"  - Circuits generated: {top_k}")
    print(f"  - Prompt strategy: {prompt_strategy}")
    print(f"  - Prompt length: {len(sys_prompt) + len(user_prompt)} chars")
    print(f"  - Knowledge base: {len(knowledge_base)} chars (FULL)")
    print(f"  - Design principles: {len(principles)} chars (FULL)")

    print(f"\nEnergy Statistics:")
    print(f"  - Best energy: {min(all_energies):.6f}")
    print(f"  - Worst energy: {max(all_energies):.6f}")
    print(f"  - Mean energy: {np.mean(all_energies):.6f}")
    print(f"  - Std energy: {np.std(all_energies):.6f}")
    print(f"  - Energy spread: {max(all_energies) - min(all_energies):.6f}")

    print(f"\nRanked circuits by energy:")
    for i, res in enumerate(sorted_results):
        print(f"  {i+1}. Energy={res['converged_energy']:.6f}, "
              f"Depth={res['depth']}, Gates={res['num_gate']}, "
              f"Topo={len(res['coupling_topology'])} couplings")

    print(f"\nOutput files:")
    print(f"  - Samples: {samples_path}")
    print(f"  - Energies: {energy_path}")
    print(f"  - Full results: {full_results_path}")
    print(f"  - Metadata: {metadata_path}")

    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()

