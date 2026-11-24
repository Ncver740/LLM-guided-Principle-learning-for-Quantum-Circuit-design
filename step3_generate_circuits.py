import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

# Add the current directory to the module search path
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
from enum import Enum
from datetime import datetime

# Import VQE training components
import tensorcircuit as tc
import tensorflow as tf
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# Set the TensorCircuit backend
tc.set_dtype("complex128")
tc.set_backend("tensorflow")


# Model type enumeration

class ModelType(Enum):
    """Enumeration of supported quantum model types."""
    HEISENBERG = "heisenberg"
    TFIM = "tfim"
    TFCM = "tfcm"


# Hyperparameter configuration

@dataclass
class Hyperparameters:
    """Centralized configuration for all hyperparameters.

    Supports experiments at different scales, such as top-10 or top-50.
    """
    # Circuit generation and evaluation
    top_k: int = 5  # Number of circuits to generate and evaluate (adjustable to 10, 50, etc.)
    num_qubits: int = 8  # Number of qubits
    max_iteration: int = 2000  # Maximum VQE iterations
    n_runs: int = 10  # Training runs per circuit (consistent with vqe_task_paralell)

    # Coupling topology
    min_coupling_topologies: int = 3  # Minimum number of distinct coupling patterns

    # Energy thresholds
    energy_improvement_threshold: float = 0.2  # Minimum energy improvement threshold
    energy_convergence_threshold: float = 1e-4  # VQE convergence criterion (0.0001, consistent with vqe_task_paralell)

    # Noise parameters
    two_qubit_depolarizing_p: float = 0.01  # Two-qubit depolarizing noise probability
    single_qubit_depolarizing_p: float = 0.001  # Single-qubit depolarizing noise probability
    bit_flip_p: float = 0.01  # Bit-flip probability

    # Prompt length limits
    max_prompt_chars: int = 50000  # Maximum prompt length in characters
    max_principles_chars: int = 3500  # Maximum principles text included in the prompt
    max_circuit_repr_chars: int = 2000  # Maximum representation length per circuit
    max_generation_attempts: int = 3  # Maximum attempts to generate valid circuits with the LLM


# Global hyperparameter instance
HP = Hyperparameters()


# API Configuration
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL = "YOUR_MODEL_NAME"

# Heisenberg model path configuration
HEISENBERG_CONFIG = {
    "principles_path": "YOUR_PATH",
    "output_dir": "YOUR_PATH",
    "samples_path": "YOUR_PATH",
    "energy_path": "YOUR_PATH"
}

# TFIM model path configuration
TFIM_CONFIG = {
    "principles_path": "YOUR_PATH",
    "output_dir": "YOUR_PATH",
    "samples_path": "YOUR_PATH",
    "energy_path": "YOUR_PATH"
}

# TFCM model path configuration
TFCM_CONFIG = {
    "principles_path": "YOUR_PATH",
    "output_dir": "YOUR_PATH",
    "samples_path": "YOUR_PATH",
    "energy_path": "YOUR_PATH"
}

# Current model configuration (can be switched through arguments)
CURRENT_CONFIG = TFCM_CONFIG  # Default model: TFIM


# Model Hamiltonian configuration

def get_hamiltonian_config(model_type: ModelType) -> Dict:
    """Get the Hamiltonian configuration for the specified model type.

    Hamiltonians for each model:

    1. Heisenberg model:
       H_Heisenberg = Σ_{i=0}^{n-1}(X_i X_{(i+1) mod n} + Y_i Y_{(i+1) mod n} + Z_i Z_{(i+1) mod n}) + Σ_{i=0}^{n-1} Z_i
       - Isotropic interactions
       - Ground-state energy (noise-free): approximately -8
       - Ground-state energy (with noise): approximately -4

    2. TFIM (transverse-field Ising model):
       H = -J Σ σ_z^i σ_z^{i+1} - h Σ σ_x^i
       - Z-direction interactions and an X-direction transverse field
       - The ground state exhibits a quantum phase transition
       - Critical point h_c/J = 1

    3. TFCM (transverse-field cluster model):
       H = -J Σ (σ_z^i σ_z^{i+1} + λ σ_x^i σ_x^{i+1}) - h Σ σ_x^i
       - Cluster interactions (including ZZ and XX terms)
       - Tunable parameter λ controls interactions within the cluster
    """
    configs = {
        ModelType.HEISENBERG: {
            "name": "Heisenberg",
            "description": "Isotropic Heisenberg model",
            "hzz": 1.0,
            "hxx": 1.0,
            "hyy": 1.0,
            "hx": 0.0,
            "hy": 0.0,
            "hz": 1.0,
            "target_energy_noisy": -11.5451,
            "target_energy_ideal": -8.0
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
            "target_energy_ideal": -7.0
        },
        ModelType.TFCM: {
            "name": "TFCM",
            "description": "Transverse Field Cluster Model",
            # TFCM (Cluster Model) Hamiltonian: H = -Σ Z_{{i-1}} X_i Z_{{i+1}} - h Σ X_i
            # This is a three-qubit interaction model, not a conventional two-qubit interaction model.
            # TensorCircuit's heisenberg_hamiltonian does not support ZXZ interactions, so equivalent parameters are used here.
            # The dataset ground-state energy for an 8-qubit system is -6.37818.
            "hzz": 0.0,
            "hxx": 1.0,
            "hyy": 0.0,
            "hx": 1.0,
            "hy": 0.0,
            "hz": 0.0,
            "target_energy_noisy": -6.37818,
            "target_energy_ideal": -7.0
        }
    }
    return configs.get(model_type, configs[ModelType.HEISENBERG])


# Prompt template configuration

def get_prompt_template(model_type: ModelType) -> Dict:
    """Get the prompt templates for the specified model type.

    Each model has dedicated system and user prompts tailored to its physical properties.
    """
    templates = {
        ModelType.HEISENBERG: {
            "system_prompt": """You design 8-qubit Heisenberg VQE circuits.
Return ONLY a Python list of circuits.
Allowed gates: rx, ry, rz, cx, cz.
Each single-qubit gate is [gate_type, qubit].
Each two-qubit gate is [gate_type, qubit1, qubit2].
Each circuit is a flat list of gates in execution order.""",

            "user_prompt_template": """Generate {num_circuits} different 8-qubit Heisenberg circuits.

Design principles:
{principles}

Requirements:
- Use exactly {num_qubits} qubits (0-7)
- Use 15-30 parameterized gates and 5-12 two-qubit gates per circuit
- Keep depth around 10-25
- Mix rx, ry, rz with cx, cz gates
- Return ONLY a Python list with no explanation

Output format: [[['rx', 0], ['cz', 0, 1], ['ry', 1], ...], [['ry', 0], ['cx', 0, 2], ...], ...]
Each circuit is a flat list where gates appear in execution order."""
        },

        ModelType.TFIM: {
            "system_prompt": """You design 8-qubit TFIM VQE circuits.
Return ONLY a Python list of circuits.
Allowed gates: rx, ry, rz, cx, cz.
Each single-qubit gate is [gate_type, qubit].
Each two-qubit gate is [gate_type, qubit1, qubit2].
Each circuit is a flat list of gates in execution order.""",

            "user_prompt_template": """Generate {num_circuits} different 8-qubit TFIM circuits.

Design principles:
{principles}

Requirements:
- Use exactly {num_qubits} qubits (0-7)
- Use 18-35 parameterized gates and 6-16 two-qubit gates per circuit
- Keep depth around 12-28
- Include enough RX rotations (TFIM needs transverse field)
- Return ONLY a Python list with no explanation

Output format: [[['rx', 0], ['cz', 0, 1], ['ry', 1], ...], [['ry', 0], ['cx', 0, 2], ...], ...]
Each circuit is a flat list where gates appear in execution order."""
        },

        ModelType.TFCM: {
            "system_prompt": """You design 8-qubit TFCM VQE circuits.
Return ONLY a Python list of circuits.
Allowed gates: rx, ry, rz, cx, cz.
Each single-qubit gate is [gate_type, qubit].
Each two-qubit gate is [gate_type, qubit1, qubit2].
Each circuit is a flat list of gates in execution order.""",

            "user_prompt_template": """Generate {num_circuits} different 8-qubit TFCM circuits.

Design principles:
{principles}

Requirements:
- Use exactly {num_qubits} qubits (0-7)
- Use 20-40 parameterized gates and 8-20 two-qubit gates per circuit
- Keep depth around 15-35
- Include enough RX rotations and cluster-like entanglement
- Return ONLY a Python list with no explanation

Output format: [[['rx', 0], ['cz', 0, 1], ['ry', 1], ...], [['ry', 0], ['cx', 0, 2], ...], ...]
Each circuit is a flat list where gates appear in execution order."""
        }
    }
    return templates.get(model_type, templates[ModelType.HEISENBERG])


# Helper functions

def load_pkl(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pkl(data, file_path):
    """Save data to a pickle file."""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def validate_file_exists(file_path, file_description):
    """Check whether a file exists."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_description} not found at {file_path}")
        return False
    return True


def load_design_principles(principles_path):
    """Load the design principles file."""
    if not validate_file_exists(principles_path, "Design principles file"):
        return get_default_principles()
    print(f"Loading design principles: {principles_path}")
    with open(principles_path, 'r', encoding='utf-8') as f:
        principles = f.read()
    print(f"Loaded design principles ({len(principles)} characters)")
    return principles


def get_default_principles():
    """Return default design principles when the file is unavailable."""
    return """## Design Principles for 8-Qubit Quantum Circuits

1. **Entanglement pattern**: Use CZ/CNOT gates to entangle neighboring qubits.
2. **Circuit depth**: Keep the depth moderate (10-25 layers) for good expressivity.
3. **Rotation diversity**: Use RX, RY, and RZ gates at roughly equal frequencies.
4. **Nearest-neighbor coupling**: Prefer two-qubit gates between neighboring qubits (e.g., 0-1, 1-2).
5. **Parameter efficiency**: Balance the number of parameterized gates against circuit depth.
"""


def compress_prompt_text(text, max_chars):
    """Limit prompt length and remove excess blank lines."""
    compact_text = re.sub(r'\n\s*\n+', '\n\n', text.strip())
    if len(compact_text) <= max_chars:
        return compact_text
    return compact_text[:max_chars].rsplit('\n', 1)[0].strip()


def check_circuit_validity(circuit, n_qubits):
    """Check whether a circuit is valid and can be trained."""
    if not isinstance(circuit, list) or len(circuit) == 0:
        return False, "Circuit must be a non-empty list"

    valid_gates = {'rx', 'ry', 'rz', 'cx', 'cz'}
    for gate in circuit:
        if not isinstance(gate, list) or len(gate) < 2:
            return False, f"Invalid gate format: {gate}"

        gate_type = gate[0]
        if gate_type not in valid_gates:
            return False, f"Invalid gate type: {gate_type}"

        if gate_type in {'rx', 'ry', 'rz'}:
            if len(gate) != 2:
                return False, f"Invalid single-qubit gate format: {gate}"
            qubit = gate[1]
            if not isinstance(qubit, int) or not 0 <= qubit < n_qubits:
                return False, f"Single-qubit gate index out of range: {gate}"
        else:
            if len(gate) != 3:
                return False, f"Invalid two-qubit gate format: {gate}"
            q1, q2 = gate[1], gate[2]
            if not isinstance(q1, int) or not isinstance(q2, int):
                return False, f"Invalid qubit indices for two-qubit gate: {gate}"
            if not (0 <= q1 < n_qubits and 0 <= q2 < n_qubits):
                return False, f"Two-qubit gate index out of range: {gate}"
            if q1 == q2:
                return False, f"A two-qubit gate cannot act on the same qubit twice: {gate}"

    return True, ""


def filter_valid_circuits(circuits, n_qubits):
    """Filter for valid, trainable circuits returned by the LLM."""
    valid_circuits = []
    for idx, circuit in enumerate(circuits, 1):
        is_valid, reason = check_circuit_validity(circuit, n_qubits)
        if not is_valid:
            print(f"Filtering out circuit {idx}: {reason}")
            continue

        if sum(1 for gate in circuit if gate[0] in {'rx', 'ry', 'rz'}) == 0:
            print(f"Filtering out circuit {idx}: no parameterized gates")
            continue

        valid_circuits.append(circuit)

    return valid_circuits


# Prompt construction functions

def build_circuit_generation_prompt(principles, num_circuits, num_qubits, model_type=ModelType.HEISENBERG):
    """Build a circuit-generation prompt for a specific model."""
    template = get_prompt_template(model_type)

    compact_principles = compress_prompt_text(principles, HP.max_principles_chars)
    system_prompt = template["system_prompt"]
    user_prompt = template["user_prompt_template"].format(
        num_circuits=num_circuits,
        num_qubits=num_qubits,
        principles=compact_principles
    )

    total_length = len(system_prompt) + len(user_prompt)
    if total_length > HP.max_prompt_chars:
        allowed_principles_chars = max(800, HP.max_principles_chars - (total_length - HP.max_prompt_chars))
        compact_principles = compress_prompt_text(principles, allowed_principles_chars)
        user_prompt = template["user_prompt_template"].format(
            num_circuits=num_circuits,
            num_qubits=num_qubits,
            principles=compact_principles
        )

    return system_prompt, user_prompt


def build_evaluation_prompt(principles, circuits_info, energies, model_type=ModelType.HEISENBERG):
    """
    Build a circuit-evaluation prompt to assess alignment with the design principles.
    """
    ham_config = get_hamiltonian_config(model_type)

    # Format circuit information
    circuit_lines = []
    for i, (info, energy) in enumerate(zip(circuits_info, energies)):
        circuit_repr = str(info)[:HP.max_circuit_repr_chars]
        circuit_lines.append(f"Circuit {i+1}: energy={energy:.6f}\nStructure: {circuit_repr}")

    circuits_text = "\n\n".join(circuit_lines)

    system_prompt = f"""You are a quantum computing expert specializing in analyzing VQE circuit design principles for the {model_type.value.upper()} model.

Your task is to analyze the relationship between quantum circuit structure and performance, and evaluate the effectiveness of the design principles.

Model information:
- Model type: {ham_config['name']} ({ham_config['description']})
- Target energy with noise: approximately {ham_config['target_energy_noisy']}
- Ideal target energy: approximately {ham_config['target_energy_ideal']}"""

    user_prompt = f"""## Task: Analyze the application of design principles in {model_type.value.upper()} quantum circuits

### Current Design Principles
{principles}

### Circuits to Analyze ({len(circuits_info)} total)
{circuits_text}

### Analysis Requirements
1. Identify structural features shared by low-energy circuits.
2. Assess how well the design principles align with actual performance.
3. Identify design principles that should be adjusted.
4. Provide specific recommendations for improvement.

### Output Format
Return a structured analysis report including:
- Successful design principles
- Principles that need improvement
- Specific recommendations for adjustment"""

    return system_prompt, user_prompt


# LLM API call function

def call_llm(system_prompt, user_prompt):
    """Call the LLM API to generate circuits or analyze them."""
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"API Base URL: {client.base_url}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    total_length = len(system_prompt) + len(user_prompt)
    print(f"\nCalling the LLM (total prompt length: {total_length} characters)...")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            n=1
        )

        result = response.choices[0].message.content
        print(f"LLM response received ({len(result)} characters)")

        # Save the LLM response to a file for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = "YOUR_PATH"
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"llm_response_{timestamp}.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SYSTEM PROMPT:\n")
            f.write("=" * 80 + "\n")
            f.write(system_prompt + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("USER PROMPT:\n")
            f.write("=" * 80 + "\n")
            f.write(user_prompt + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("LLM RESPONSE:\n")
            f.write("=" * 80 + "\n")
            f.write(result + "\n")
        print(f"LLM response saved to: {debug_file}")

        return result

    except Exception as e:
        print(f"LLM API call failed: {e}")
        # Save the error details
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = "YOUR_PATH"
        os.makedirs(debug_dir, exist_ok=True)
        error_file = os.path.join(debug_dir, f"llm_error_{timestamp}.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR:\n")
            f.write("=" * 80 + "\n")
            f.write(str(e) + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("SYSTEM PROMPT:\n")
            f.write("=" * 80 + "\n")
            f.write(system_prompt + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("USER PROMPT:\n")
            f.write("=" * 80 + "\n")
            f.write(user_prompt + "\n")
        print(f"Error details saved to: {error_file}")
        raise


def parse_generated_circuits(response):
    """Parse the circuit list returned by the LLM, with debugging and automatic repair of excess nesting."""
    try:
        # Save the raw response for debugging
        print("\n" + "=" * 80)
        print("Raw LLM response (first 1000 characters):")
        print("=" * 80)
        print(response[:1000])
        print("=" * 80 + "\n")

        # Remove Markdown code fences
        response = re.sub(r'```python\s*', '', response)
        response = re.sub(r'```\s*', '', response)

        lines = response.strip().split('\n')
        circuit_text = None

        # Find the start of the list
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
            # Remove leading whitespace to avoid an "unexpected indent" error
            circuit_text = circuit_text.strip()

            print(f"Extracted circuit text (first 500 characters):\n{circuit_text[:500]}\n")
            circuits = ast.literal_eval(circuit_text)
            print(f"Successfully parsed {len(circuits)} circuits")

            # Show the structure of each circuit
            for idx, circuit in enumerate(circuits[:3], 1):  # Show only the first 3
                print(f"  Circuit {idx} structure: {type(circuit)}, length={len(circuit) if isinstance(circuit, list) else 'N/A'}")
                if isinstance(circuit, list) and len(circuit) > 0:
                    print(f"    First element: {circuit[0]}, type={type(circuit[0])}")

            # Automatically repair excess nesting
            # For example, flatten [[[['rx', 0]]]] to [['rx', 0]]
            fixed_circuits = []
            for circuit_idx, circuit in enumerate(circuits):
                if not isinstance(circuit, list):
                    print(f"  Warning: circuit {circuit_idx+1} is not a list; skipping")
                    continue

                fixed_circuit = []
                for gate_idx, gate in enumerate(circuit):
                    # Check for excess nesting: [[['rx', 0]]] or [['rx', 0]]
                    if isinstance(gate, list):
                        # If gate is a list, inspect its contents
                        if len(gate) == 1 and isinstance(gate[0], list):
                            # Excess nesting: [[['rx', 0]]] -> ['rx', 0]
                            inner = gate[0]
                            if len(inner) >= 2:
                                fixed_circuit.append(inner)
                            else:
                                print(f"  Warning: gate {gate_idx+1} in circuit {circuit_idx+1} has an invalid format: {gate}")
                        elif len(gate) >= 2 and isinstance(gate[0], str):
                            # Expected format: ['rx', 0]
                            fixed_circuit.append(gate)
                        else:
                            print(f"  Warning: gate {gate_idx+1} in circuit {circuit_idx+1} has an invalid format: {gate}")
                    else:
                        print(f"  Warning: gate {gate_idx+1} in circuit {circuit_idx+1} is not a list: {gate}")

                if fixed_circuit:
                    fixed_circuits.append(fixed_circuit)

            if len(fixed_circuits) != len(circuits):
                print(f"\nAutomatic repair: extracted {len(fixed_circuits)} valid circuits from {len(circuits)} circuits")

            # Show the repaired structure
            print("\nRepaired circuit structure:")
            for idx, circuit in enumerate(fixed_circuits[:3], 1):
                print(f"  Circuit {idx}: {len(circuit)} gates")
                if len(circuit) > 0:
                    print(f"    Example gates: {circuit[0]}, {circuit[1] if len(circuit) > 1 else 'N/A'}")

            return fixed_circuits
        else:
            print("Could not parse circuits from the response")
            print(f"Full response:\n{response}\n")
            return []

    except Exception as e:
        print(f"Error parsing circuits: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        print(f"Full response:\n{response}\n")
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


def calculate_circuit_stats(circuit):
    """Calculate circuit statistics."""
    depth = 0
    num_gate = len(circuit)
    num_2q_gate = 0

    for gate in circuit:
        if len(gate) == 3:
            num_2q_gate += 1

    return depth, num_gate, num_2q_gate


# VQE trainer class

class VQETrainer:
    """VQE trainer supporting multiple Hamiltonians and full noise simulation.

    Supported models:
    - Heisenberg
    - TFIM (transverse-field Ising model)
    - TFCM (transverse-field cluster model)
    """

    def __init__(self, n_qubit, model_type=ModelType.HEISENBERG, max_iteration=2000, n_runs=5,
                 noise=True, two_qubit_depolarizing_p=0.01,
                 single_qubit_depolarizing_p=0.001, bit_flip_p=0.01):
        """Initialize the VQE trainer.

        Args:
            n_qubit: Number of qubits.
            model_type: Model type (Heisenberg/TFIM/TFCM).
            max_iteration: Maximum number of iterations.
            n_runs: Number of training runs per circuit.
            noise: Whether to enable noise.
            two_qubit_depolarizing_p: Two-qubit depolarizing noise probability.
            single_qubit_depolarizing_p: Single-qubit depolarizing noise probability.
            bit_flip_p: Bit-flip probability.
        """
        self.n_qubit = n_qubit
        self.model_type = model_type
        self.max_iteration = max_iteration
        self.n_runs = n_runs
        self.noise = noise

        # Get the Hamiltonian configuration
        ham_config = get_hamiltonian_config(model_type)

        # Set the backend
        self.K = tc.set_backend("tensorflow")

        # Create the Hamiltonian for the selected model
        if model_type == ModelType.TFCM:
            # TFCM uses a custom transverse-field cluster-model Hamiltonian
            self.h = self._build_tfcm_hamiltonian(n_qubit)
            print("VQE trainer initialized:")
            print(f"  - Model type: {ham_config['name']} (transverse-field cluster model)")
            print(f"  - Number of qubits: {n_qubit}")
            print(f"  - Hamiltonian shape: {self.h.shape}")
        else:
            # Heisenberg and TFIM use standard Hamiltonians
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

        print("VQE trainer initialized:")
        print(f"  - Model type: {ham_config['name']} ({ham_config['description']})")
        print(f"  - Number of qubits: {n_qubit}")
        print(f"  - Target energy with noise: approximately {ham_config['target_energy_noisy']}")
        print(f"  - Ideal target energy: approximately {ham_config['target_energy_ideal']}")

        # Configure noise parameters
        if self.noise:
            self.two_qubit_channel_depolarizing_p = two_qubit_depolarizing_p
            self.single_qubit_channel_depolarizing_p = single_qubit_depolarizing_p
            self.bit_flip_p = bit_flip_p

            # Configure the two-qubit depolarizing channel
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.two_qubit_channel_depolarizing_p / 15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)

            # Configure the single-qubit depolarizing channel
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.single_qubit_channel_depolarizing_p / 3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

            print("Noise enabled:")
            print(f"  - Two-qubit depolarizing p: {two_qubit_depolarizing_p}")
            print(f"  - Single-qubit depolarizing p: {single_qubit_depolarizing_p}")
            print(f"  - Bit-flip p: {bit_flip_p}")
        else:
            print("Noise disabled (ideal simulation)")


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

    def get_param_num(self, cir):
        """Count the circuit parameters."""
        param_num = 0
        for gate in cir:
            if gate[0] in ['rx', 'ry', 'rz']:
                param_num += 1
        return param_num

    def compute_energy(self, param, structure):
        """Calculate circuit energy, with optional noise simulation."""
        if self.noise:
            # Define K0 and K1 matrices for bit-flip noise
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            # Create the quantum circuit in density-matrix form
            c = tc.DMCircuit(self.n_qubit)
            param_index = 0

            # Iterate through the circuit structure
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

            # Add bit-flip noise
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            # Calculate the energy
            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)

        else:
            # Create the quantum circuit
            c = tc.Circuit(self.n_qubit)
            param_index = 0

            # Iterate through the circuit structure
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

            # Calculate the energy
            e = tc.templates.measurements.operator_expectation(c, self.h)

        return e

    def train_circuit(self, circuit, seed=42):
        """Train a single circuit."""
        np.random.seed(seed)
        tf.random.set_seed(seed)

        param_num = self.get_param_num(circuit)
        if param_num == 0:
            print("Warning: circuit has no parameterized gates!")
            return float('inf'), None, []

        # Define the trainer
        trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0))

        # Initialize parameters using depth_count, consistent with vqe_task_paralell and test_xl.py
        L = depth_count(circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1/(8*(L+2)), size=param_num)
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )

        # Optimizer
        opt = tf.keras.optimizers.Adam(0.05)

        # Train
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

    def train_multiple_runs(self, circuit):
        """Run multiple training trials and return the best result."""
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

            print(f"  Run {run_id+1}/{self.n_runs}: energy = {energy:.6f}")

        return best_energy, best_params, best_energy_epoch


def format_circuit_output(circuit):
    """Format circuit output in samples.pkl format."""
    depth, num_gate, num_2q_gate = calculate_circuit_stats(circuit)

    # Calculate depth using a simplified sequential count
    depth = 0
    current_layer_ops = []
    for gate in circuit:
        if len(gate) == 3:  # Two-qubit gate
            if current_layer_ops:
                depth += 1
                current_layer_ops = []
            depth += 1
        else:  # Single-qubit gate
            current_layer_ops.append(gate)
    if current_layer_ops:
        depth += 1

    return [circuit, depth, num_gate, num_2q_gate]


def create_default_circuits(num_circuits=5, num_qubits=8):
    """Create default circuits if LLM generation fails."""
    circuits = []

    for _ in range(num_circuits):
        circuit = []
        # Basic layered pattern
        for layer in range(12):
            for q in range(num_qubits - 1):
                gate_type = random.choice(['rx', 'ry', 'rz'])
                circuit.append([gate_type, q])
            if layer % 2 == 0:
                circuit.append(['cz', layer % (num_qubits - 1), (layer + 1) % (num_qubits - 1)])
            else:
                circuit.append(['cx', layer % (num_qubits - 1), (layer + 1) % (num_qubits - 1)])

        circuits.append(circuit)

    return circuits


# Main function

def main():
    """Run the main workflow."""
    parser = argparse.ArgumentParser(description='Quantum circuit design principle learning v25 - multi-model VQE')

    # Select the model type
    parser.add_argument("--model", type=str, default="heisenberg",
                       choices=["heisenberg", "tfim", "tfcm"],
                       help="Select a model type: heisenberg, tfim, or tfcm")

    # Configure hyperparameters
    parser.add_argument("--top_k", type=int, default=HP.top_k,
                       help=f"Number of circuits to generate and evaluate (default: {HP.top_k})")
    parser.add_argument("--num_qubits", type=int, default=HP.num_qubits,
                       help=f"Number of qubits (default: {HP.num_qubits})")

    # Noise parameters
    parser.add_argument("--noise", type=bool, default=True,
                       help="Whether to enable noise simulation")
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

    # Resolve the model type
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

    # Get the Hamiltonian configuration
    ham_config = get_hamiltonian_config(model_type)

    print("=" * 80)
    print("Quantum Circuit Design Principle Learning v25")
    print("Multi-Model VQE Circuit Generation")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  - Model type: {ham_config['name']} ({ham_config['description']})")
    print(f"  - TOP_K: {args.top_k}")
    print(f"  - Number of qubits: {args.num_qubits}")
    print(f"  - Noise enabled: {args.noise}")
    print(f"  - Output directory: {config['output_dir']}")
    print("=" * 80)

    # Update global hyperparameters
    HP.top_k = args.top_k
    HP.num_qubits = args.num_qubits

    # Step 1: Load design principles
    print("\n[Step 1] Loading design principles...")
    principles = load_design_principles(config["principles_path"])

    # Step 2: Generate circuits with the LLM
    print(f"\n[Step 2] Generating {args.model.upper()} quantum circuits with the LLM...")
    circuits = []
    seen_circuits = set()
    sys_prompt, user_prompt = build_circuit_generation_prompt(
        principles,
        num_circuits=args.top_k,
        num_qubits=args.num_qubits,
        model_type=model_type
    )

    for attempt in range(HP.max_generation_attempts):
        remaining = args.top_k - len(circuits)
        if remaining <= 0:
            break

        print(f"LLM generation attempt {attempt + 1}/{HP.max_generation_attempts}; {remaining} more circuits needed")
        sys_prompt, user_prompt = build_circuit_generation_prompt(
            principles,
            num_circuits=remaining,
            num_qubits=args.num_qubits,
            model_type=model_type
        )
        llm_response = call_llm(sys_prompt, user_prompt)
        parsed_circuits = parse_generated_circuits(llm_response)
        valid_circuits = filter_valid_circuits(parsed_circuits, args.num_qubits)

        for circuit in valid_circuits:
            circuit_key = tuple(tuple(gate) for gate in circuit)
            if circuit_key in seen_circuits:
                continue
            seen_circuits.add(circuit_key)
            circuits.append(circuit)
            if len(circuits) >= args.top_k:
                break

    if not circuits:
        print("\n" + "=" * 80)
        print("Error: the LLM did not generate any valid, trainable circuits")
        print("=" * 80)
        print("\nCheck the debug files in the configured debug directory for details:")
        print("  - llm_response_*.txt: Full LLM response")
        print("  - llm_error_*.txt: API call error details")
        print("\nPossible causes:")
        print("  1. The LLM response format is unexpected (excess nesting, missing elements, etc.)")
        print("  2. The generated circuits do not pass validation (too few parameterized or two-qubit gates, etc.)")
        print("  3. The API call failed or timed out")
        print("=" * 80 + "\n")
        raise RuntimeError("The LLM did not generate any valid, trainable circuits. Training stopped; see the debug files.")

    circuits = circuits[:args.top_k]
    print(f"Generated {len(circuits)} valid LLM circuits")

    # Step 3: Train circuits with VQE
    print(f"\n[Step 3] Training circuits with VQE (noise {'enabled' if args.noise else 'disabled'})...")
    trainer = VQETrainer(
        n_qubit=args.num_qubits,
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
        print(f"\n--- Training circuit {i+1}/{len(circuits)} ---")
        print(f"Circuit length: {len(circuit)} gates")

        # Train
        best_energy, best_params, energy_epoch = trainer.train_multiple_runs(circuit)

        # Format the output
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
            'noise_params': {
                'noise': args.noise,
                'two_qubit_depolarizing_p': args.two_qubit_depolarizing_p,
                'single_qubit_depolarizing_p': args.single_qubit_depolarizing_p,
                'bit_flip_p': args.bit_flip_p
            }
        }

        results.append(result)
        all_energies.append(best_energy)

        print(f"Circuit {i+1} converged energy: {best_energy:.6f}")

    # Step 4: Save results
    print("\n[Step 4] Saving results...")

    # Create a timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save in pickle format
    samples_output = [r['circuit'] for r in results]
    samples_path = os.path.join(output_dir, "generated_samples.pkl")
    save_pkl(samples_output, samples_path)

    # Save energies
    energy_path = os.path.join(output_dir, "generated_energy.pkl")
    save_pkl(all_energies, energy_path)

    # Save the full results
    full_results_path = os.path.join(output_dir, "full_results.pkl")
    save_pkl(results, full_results_path)

    # Save metadata
    metadata = {
        'model_type': model_type.value,
        'model_name': ham_config['name'],
        'top_k': args.top_k,
        'timestamp': timestamp,
        'noise_params': {
            'noise': args.noise,
            'two_qubit_depolarizing_p': args.two_qubit_depolarizing_p,
            'single_qubit_depolarizing_p': args.single_qubit_depolarizing_p,
            'bit_flip_p': args.bit_flip_p
        }
    }
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    save_pkl(metadata, metadata_path)

    # Step 5: Print a summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)

    print("\nModel configuration:")
    print(f"  - Model type: {ham_config['name']}")
    print(f"  - Target energy with noise: approximately {ham_config['target_energy_noisy']}")

    print("\nNoise configuration:")
    print(f"  - Noise enabled: {args.noise}")
    if args.noise:
        print(f"  - Two-qubit depolarizing p: {args.two_qubit_depolarizing_p}")
        print(f"  - Single-qubit depolarizing p: {args.single_qubit_depolarizing_p}")
        print(f"  - Bit-flip p: {args.bit_flip_p}")

    print(f"\nNumber of circuits generated: {len(circuits)}")

    # Sort by ascending energy
    sorted_results = sorted(results, key=lambda x: x['converged_energy'])

    print("\nCircuit details:")
    for i, result in enumerate(sorted_results):
        print(f"  Circuit {i+1}: depth={result['depth']}, gates={result['num_gate']}, "
              f"two-qubit gates={result['num_2q_gate']}, energy={result['converged_energy']:.6f}")

    print("\nEnergy statistics:")
    print(f"  Best: {min(all_energies):.6f}")
    print(f"  Worst: {max(all_energies):.6f}")
    print(f"  Mean: {np.mean(all_energies):.6f}")
    print(f"  Standard deviation: {np.std(all_energies):.6f}")

    print("\nOutput files:")
    print(f"  - Circuit samples: {samples_path}")
    print(f"  - Energy data: {energy_path}")
    print(f"  - Full results: {full_results_path}")

    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
