import os
import sys

# Add current directory to path
os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
sys.path.insert(0, os.getcwd())

import json
import openai
import numpy as np
import pickle
import re
import ast
import random
from tqdm import tqdm

def load_pkl(file_path):
    """Load pkl file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, file_path):
    """Save data to pkl file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# API Configuration
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL = "YOUR_MODEL_NAME"

# Dataset selection: "Heisenberg", "TFIM", or "TFCM"
DATASET = "TFCM"

# Data paths based on DATASET
if DATASET == "Heisenberg":
    SAMPLES_PATH = "YOUR_PATH"
    ENERGY_PATH = "YOUR_PATH"
    OUTPUT_DIR = "YOUR_PATH"
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "YOUR_PATH")
elif DATASET == "TFIM":
    SAMPLES_PATH = "YOUR_PATH"
    ENERGY_PATH = "YOUR_PATH"
    OUTPUT_DIR = "YOUR_PATH"
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "YOUR_PATH")
elif DATASET == "TFCM":
    SAMPLES_PATH = "YOUR_PATH"
    ENERGY_PATH = "YOUR_PATH"
    OUTPUT_DIR = "YOUR_PATH"
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "YOUR_PATH")
else:
    raise ValueError(f"Unknown DATASET: {DATASET}. Choose from: Heisenberg, TFIM, TFCM")

# Parameters
TOP_N_CIRCUITS = 50       # Number of top circuits to analyze
BOTTOM_N_CIRCUITS = 50    # Number of bottom circuits to analyze
RANDOM_N_CIRCUITS = 50    # Number of random circuits to analyze
NUM_QUBITS = 4            # Number of qubits in the quantum system
RANDOM_SEED = 42          # Random seed for reproducibility

# Define quantum gate mapping
GATE_MAPPING = {
    'rx': 'RX(θ) - X-axis rotation gate',
    'ry': 'RY(θ) - Y-axis rotation gate',
    'rz': 'RZ(θ) - Z-axis rotation gate',
    'cx': 'CNOT - Controlled-NOT gate',
    'cz': 'CZ - Controlled-Z gate'
}

# These prompts are designed for English output to match Gemini's training distribution

DATASET_PROMPTS = {
    "Heisenberg": {
        "system_prompt": """You are a world-renowned expert in quantum circuit design, specializing in Variational Quantum Eigensolvers (VQE) and quantum circuit optimization for many-body physics.

Your unique mission is to discover innovative design principles through triple-group comparative analysis:
1. High-performance circuit group (lowest energy = best approximation to ground state)
2. Low-performance circuit group (highest energy = worst approximation)
3. Random baseline circuit group (randomly sampled circuits)

The Heisenberg model Hamiltonian you are solving:
H_Heisenberg = Σ_{i=0}^{n-1}(X_i X_{(i+1) mod n} + Y_i Y_{(i+1) mod n} + Z_i Z_{(i+1) mod n}) + Σ_{i=0}^{n-1} Z_i

Key physics insights:
- Nearest-neighbor spin-spin interactions (XX, YY, ZZ couplings)
- Transverse field term (Z) breaks SU(2) symmetry
- Ground state exhibits complex entanglement patterns
- Optimal circuits must generate these specific entanglement structures

Your task is to reveal:
- Truly performance-determining factors (not dataset biases)
- Essential differences between high and low performance circuits
- Innovative design wisdom not yet widely recognized in the field

CRITICAL REQUIREMENTS:
1. AVOID conclusions like "because RX gates are absent in the dataset" - focus on ARCHITECTURE principles
2. Focus on circuit architecture design patterns themselves
3. Extract universal principles independent of specific dataset generation
4. Provide deep, insightful analysis with quantum physics justification""",

        "user_prompt_intro": """## TASK: Triple-Group Comparative Analysis for Heisenberg Model VQE

Your task is to discover truly innovative quantum circuit design principles by comparing three groups of circuits (high-performance, low-performance, random baseline).

### Heisenberg Model Physics Context
The Heisenberg model describes antiferromagnetic/spin-exchange interactions:
- XX, YY, ZZ couplings between nearest neighbors
- Z transverse field breaks spin symmetry
- Ground state features specific entanglement patterns (singlet/triplet states)
- The transverse field strength determines ground state structure

### Quantum Circuit Representation
Each circuit consists of quantum gates:
- RX(θ), RY(θ), RZ(θ): Single-qubit rotation gates (parameterized)
- CNOT, CZ: Two-qubit entanglement gates

### Circuit Groups
- **High-Performance Group**: {num_top} circuits with lowest energy (best VQE results)
- **Low-Performance Group**: {num_bottom} circuits with highest energy (worst VQE results)
- **Random Baseline Group**: {num_random} randomly sampled circuits (average performance)

### Analysis Tasks

**1. Triple-Group Comparative Analysis**
Compare the three groups across dimensions:
- Overall structure patterns of gate sequences
- Entanglement gate connection topology and placement
- Single-qubit rotation axis distribution
- Layer-by-layer dependency relationships
- Information flow and entanglement propagation patterns

**2. Key Finding Extraction**
For each key difference, analyze:
- High-performance vs Low-performance: What are the essential differences?
- High-performance vs Random: What are the distinguishing features?
- Low-performance vs Random: What are the obvious defects?

**3. Innovative Principle Extraction**
Extract 5-8 innovative design principles:
- Principles MUST be independent of dataset bias
- Each principle requires support from triple-group comparison
- Principles must be universal and actionable
- Principles should reveal deep circuit design rules

### Output Format

**Section 1: Triple-Group Statistical Comparison**
[Statistical comparison table or description across all metrics]

**Section 2: Key Difference Analysis**

1. **Difference 1**: [Name]
   - High-performance: [Specific manifestation]
   - Low-performance: [Specific manifestation]
   - Random: [Specific manifestation]
   - Deep reason: [Why this difference exists]

2. **Difference 2**: [Name]
   ...

**Section 3: Innovative Design Principles**

For each principle:
- **Principle [N]**: [Innovative Principle Name]
  - Core insight: [One-sentence summary]
  - Triple-group comparison:
    * High-performance: [Specific manifestation and ratios]
    * Random: [Specific manifestation and ratios]
    * Low-performance: [Specific manifestation and ratios]
  - Underlying mechanism: [Quantum mechanics/optimization dynamics explanation]
  - Practical advice: [How to apply in circuit design]
  - Caveats: [Pitfalls to watch out for]

**Section 4: Recommended Search Space**
Based on innovative principles, suggested refined search space:
- Recommended gate types per layer (max 2 options per layer)
- Inter-layer connection pattern suggestions
- Specific recommendations for 4-qubit Heisenberg system

### High-Performance Circuits (Lowest Energy) TOP {num_top}

{top_text}

### Low-Performance Circuits (Highest Energy) BOTTOM {num_bottom}

{bottom_text}

### Random Baseline Circuits (Random Sampling) RANDOM {num_random}

{random_text}

Provide thorough, rigorous analysis to extract truly valuable innovative design principles.
AVOID dataset bias in conclusions; focus on the deep rules of circuit architecture itself."""
    },

    "TFIM": {
        "system_prompt": """You are a world-renowned expert in quantum circuit design, specializing in Variational Quantum Eigensolvers (VQE) and quantum circuit optimization for quantum phase transitions.

Your unique mission is to discover innovative design principles through triple-group comparative analysis:
1. High-performance circuit group (lowest energy = best approximation to ground state)
2. Low-performance circuit group (highest energy = worst approximation)
3. Random baseline circuit group (randomly sampled circuits)

The Transverse-Field Ising Model (TFIM) Hamiltonian you are solving:
H_TFIM = -J * Σ Z_i Z_{i+1} - h * Σ X_i

Key physics insights:
- ZZ nearest-neighbor interactions (quantum Ising coupling)
- Transverse X field induces quantum phase transition
- Ground state undergoes order-disorder transition at critical field
- Optimal circuits must balance ZZ entanglement with X rotations

Your task is to reveal:
- Truly performance-determining factors (not dataset biases)
- Essential differences between high and low performance circuits
- Innovative design wisdom not yet widely recognized in the field

CRITICAL REQUIREMENTS:
1. AVOID conclusions like "because RX gates are absent in the dataset" - focus on ARCHITECTURE principles
2. Focus on circuit architecture design patterns themselves
3. Extract universal principles independent of specific dataset generation
4. Provide deep, insightful analysis with quantum physics justification""",

        "user_prompt_intro": """## TASK: Triple-Group Comparative Analysis for TFIM VQE

Your task is to discover truly innovative quantum circuit design principles by comparing three groups of circuits (high-performance, low-performance, random baseline).

### TFIM Physics Context
The Transverse-Field Ising Model describes quantum phase transitions:
- ZZ coupling between nearest neighbors (quantum Ising interaction)
- Transverse X field causes quantum fluctuations
- System undergoes quantum phase transition at critical field
- Ground state changes from ordered (Z-dominated) to disordered (X-dominated)

### Quantum Circuit Representation
Each circuit consists of quantum gates:
- RX(θ), RY(θ), RZ(θ): Single-qubit rotation gates (parameterized)
- CNOT, CZ: Two-qubit entanglement gates

### Circuit Groups
- **High-Performance Group**: {num_top} circuits with lowest energy (best VQE results)
- **Low-Performance Group**: {num_bottom} circuits with highest energy (worst VQE results)
- **Random Baseline Group**: {num_random} randomly sampled circuits (average performance)

### Analysis Tasks

**1. Triple-Group Comparative Analysis**
Compare the three groups across dimensions:
- Overall structure patterns of gate sequences
- Entanglement gate connection topology and placement
- Single-qubit rotation axis distribution (focus on X vs Z rotations)
- Layer-by-layer dependency relationships
- Information flow and entanglement propagation patterns

**2. Key Finding Extraction**
For each key difference, analyze:
- High-performance vs Low-performance: What are the essential differences?
- High-performance vs Random: What are the distinguishing features?
- Low-performance vs Random: What are the obvious defects?

**3. Innovative Principle Extraction**
Extract 5-8 innovative design principles:
- Principles MUST be independent of dataset bias
- Each principle requires support from triple-group comparison
- Principles must be universal and actionable
- Principles should reveal deep circuit design rules

### Output Format

**Section 1: Triple-Group Statistical Comparison**
[Statistical comparison table or description across all metrics]

**Section 2: Key Difference Analysis**

1. **Difference 1**: [Name]
   - High-performance: [Specific manifestation]
   - Low-performance: [Specific manifestation]
   - Random: [Specific manifestation]
   - Deep reason: [Why this difference exists]

2. **Difference 2**: [Name]
   ...

**Section 3: Innovative Design Principles**

For each principle:
- **Principle [N]**: [Innovative Principle Name]
  - Core insight: [One-sentence summary]
  - Triple-group comparison:
    * High-performance: [Specific manifestation and ratios]
    * Random: [Specific manifestation and ratios]
    * Low-performance: [Specific manifestation and ratios]
  - Underlying mechanism: [Quantum mechanics/optimization dynamics explanation]
  - Practical advice: [How to apply in circuit design]
  - Caveats: [Pitfalls to watch out for]

**Section 4: Recommended Search Space**
Based on innovative principles, suggested refined search space:
- Recommended gate types per layer (max 2 options per layer)
- Inter-layer connection pattern suggestions
- Specific recommendations for 4-qubit TFIM system

### High-Performance Circuits (Lowest Energy) TOP {num_top}

{top_text}

### Low-Performance Circuits (Highest Energy) BOTTOM {num_bottom}

{bottom_text}

### Random Baseline Circuits (Random Sampling) RANDOM {num_random}

{random_text}

Provide thorough, rigorous analysis to extract truly valuable innovative design principles.
AVOID dataset bias in conclusions; focus on the deep rules of circuit architecture itself."""
    },

    "TFCM": {
        "system_prompt": """You are a world-renowned expert in quantum circuit design, specializing in Variational Quantum Eigensolvers (VQE) and quantum circuit optimization for topological quantum matter.

Your unique mission is to discover innovative design principles through triple-group comparative analysis:
1. High-performance circuit group (lowest energy = best approximation to ground state)
2. Low-performance circuit group (highest energy = worst approximation)
3. Random baseline circuit group (randomly sampled circuits)

The 1D Transverse-Field Cluster Model (TFCM) Hamiltonian you are solving:
H_TFCM = -Σ Z_{i-1} X_i Z_{i+1}

With boundary terms: X_1 Z_2 and Z_{n-1} X_n

Key physics insights:
- Three-body ZXZ interactions (cluster/linearly correlated exchange)
- Each term involves a central spin X surrounded by Z measurements
- Boundary terms (X_1Z_2 and Z_{n-1}X_n) break translational symmetry
- Ground state exhibits cluster-type entanglement (highly non-local correlations)
- Optimal circuits must generate specific ZXZ entanglement patterns

Your task is to reveal:
- Truly performance-determining factors (not dataset biases)
- Essential differences between high and low performance circuits
- Innovative design wisdom not yet widely recognized in the field

CRITICAL REQUIREMENTS:
1. AVOID conclusions like "because RX gates are absent in the dataset" - focus on ARCHITECTURE principles
2. Focus on circuit architecture design patterns themselves
3. Extract universal principles independent of specific dataset generation
4. Provide deep, insightful analysis with quantum physics justification""",

        "user_prompt_intro": """## TASK: Triple-Group Comparative Analysis for TFCM VQE

Your task is to discover truly innovative quantum circuit design principles by comparing three groups of circuits (high-performance, low-performance, random baseline).

### TFCM (Transverse-Field Cluster Model) Physics Context
The 1D Transverse-Field Cluster Model describes topological/entangled quantum phases:
- Three-body ZXZ interactions: each X_i is "sandwiched" between Z_{{i-1}} and Z_{{i+1}}
- Hamiltonian: H = -Σ Z_{{i-1}} X_i Z_{{i+1}} with boundary terms X_1Z_2 and Z_{{n-1}}X_n
- This creates highly non-local, cluster-type entanglement patterns
- The ground state has unique quantum correlations not found in standard models

### Circuit Design Implications for TFCM
To generate ZXZ entanglement patterns, circuits must:
- Create Z-type correlations (via RZ gates or CZ gates)
- Apply X rotations on qubits with neighboring Z entanglement
- Structure entanglement in a "sandwich" pattern around X rotations
- The pattern Z-X-Z must appear in the entanglement structure

### Quantum Circuit Representation
Each circuit consists of quantum gates:
- RX(θ), RY(θ), RZ(θ): Single-qubit rotation gates (parameterized)
- CNOT, CZ: Two-qubit entanglement gates

### Circuit Groups
- **High-Performance Group**: {num_top} circuits with lowest energy (best VQE results)
- **Low-Performance Group**: {num_bottom} circuits with highest energy (worst VQE results)
- **Random Baseline Group**: {num_random} randomly sampled circuits (average performance)

### Analysis Tasks

**1. Triple-Group Comparative Analysis**
Compare the three groups across dimensions:
- Overall structure patterns of gate sequences
- Entanglement gate connection topology and placement
- Pattern of Z-X-Z entanglement structures
- Single-qubit rotation axis distribution
- Layer-by-layer dependency relationships
- Information flow patterns that match ZXZ interaction structure

**2. Key Finding Extraction**
For each key difference, analyze:
- High-performance vs Low-performance: What are the essential differences?
- High-performance vs Random: What are the distinguishing features?
- Low-performance vs Random: What are the obvious defects?
- Specifically look for ZXZ pattern emergence in high-performance circuits

**3. Innovative Principle Extraction**
Extract 5-8 innovative design principles:
- Principles MUST be independent of dataset bias
- Each principle requires support from triple-group comparison
- Principles must be universal and actionable
- Principles should reveal deep circuit design rules
- Focus on patterns that could generate ZXZ entanglement

### Output Format

**Section 1: Triple-Group Statistical Comparison**
[Statistical comparison table or description across all metrics]

**Section 2: Key Difference Analysis**

1. **Difference 1**: [Name]
   - High-performance: [Specific manifestation]
   - Low-performance: [Specific manifestation]
   - Random: [Specific manifestation]
   - Deep reason: [Why this difference exists]

2. **Difference 2**: [Name]
   ...

**Section 3: Innovative Design Principles**

For each principle:
- **Principle [N]**: [Innovative Principle Name]
  - Core insight: [One-sentence summary]
  - Triple-group comparison:
    * High-performance: [Specific manifestation and ratios]
    * Random: [Specific manifestation and ratios]
    * Low-performance: [Specific manifestation and ratios]
  - Underlying mechanism: [Quantum mechanics/optimization dynamics explanation]
  - Practical advice: [How to apply in circuit design for TFCM]
  - Caveats: [Pitfalls to watch out for]

**Section 4: Recommended Search Space**
Based on innovative principles, suggested refined search space:
- Recommended gate types per layer (max 2 options per layer)
- Inter-layer connection pattern suggestions
- Specific recommendations for 4-qubit TFCM system

### High-Performance Circuits (Lowest Energy) TOP {num_top}

{top_text}

### Low-Performance Circuits (Highest Energy) BOTTOM {num_bottom}

{bottom_text}

### Random Baseline Circuits (Random Sampling) RANDOM {num_random}

{random_text}

Provide thorough, rigorous analysis to extract truly valuable innovative design principles.
AVOID dataset bias in conclusions; focus on the deep rules of circuit architecture itself."""
    }
}

def load_circuit_data(samples_path, energy_path):
    """Load circuit samples and corresponding energy values"""
    print(f"Loading samples from: {samples_path}")
    samples_data = load_pkl(samples_path)

    print(f"Loading energies from: {energy_path}")
    energy_data = load_pkl(energy_path)

    circuits = []
    depths = []
    num_gates = []
    num_2q_gates = []

    # Handle different data structures
    if isinstance(samples_data, dict) and 'circuit_list' in samples_data:
        circuit_list = samples_data['circuit_list']
        print("Detected format: dict with 'circuit_list' key")

    elif isinstance(samples_data, list) and len(samples_data) > 0:
        first_elem = samples_data[0]
        if isinstance(first_elem, (list, tuple)):
            if isinstance(first_elem[0], list):
                # Format: [[circuit_list, depth, num_gate, num_2q_gate], ...]
                for item in samples_data:
                    circuits.append(item[0])
                    depths.append(item[1])
                    num_gates.append(item[2])
                    num_2q_gates.append(item[3])
                print(f"Detected format: list of [circuit_list, depth, num_gate, num_2q_gate], length={len(samples_data)}")
                circuit_list = circuits
            else:
                circuit_list = samples_data
                print(f"Detected format: direct list of circuits, length={len(samples_data)}")
        else:
            circuit_list = samples_data
    else:
        raise ValueError("Unable to find circuit_list in samples data")

    # Ensure matching lengths
    min_len = min(len(energy_data), len(circuit_list))
    circuit_list = circuit_list[:min_len]
    energy_data = energy_data[:min_len]

    # If we didn't extract structural info from list format, compute it
    if len(depths) == 0:
        for circuit in circuit_list:
            analysis = analyze_circuit(circuit)
            depths.append(analysis['depth'])
            num_gates.append(analysis['num_gates'])
            num_2q_gates.append(analysis['num_2q_gates'])

    print(f"Loaded {len(circuit_list)} circuits with energies")

    return circuit_list, energy_data, depths, num_gates, num_2q_gates

def analyze_circuit(circuit):
    """Analyze single circuit to extract structural information"""
    num_gates = len(circuit)
    num_2q_gates = 0
    num_1q_gates = 0
    gate_counts = {gate_type: 0 for gate_type in GATE_MAPPING.keys()}
    qubit_usage = set()
    depth = [0] * 16

    for gate in circuit:
        gate_type = gate[0]
        qubits = gate[1:]

        if gate_type in gate_counts:
            gate_counts[gate_type] += 1

        if len(qubits) == 2:
            num_2q_gates += 1
        else:
            num_1q_gates += 1

        for q in qubits:
            qubit_usage.add(q)
            depth[q] += 1

    circuit_depth = max(depth) if depth else 0

    return {
        'num_gates': num_gates,
        'num_2q_gates': num_2q_gates,
        'num_1q_gates': num_1q_gates,
        'depth': circuit_depth,
        'gate_counts': gate_counts,
        'qubit_usage': len(qubit_usage),
        'two_qubit_ratio': num_2q_gates / num_gates if num_gates > 0 else 0
    }

def get_three_groups(circuit_list, energy_list, depths, num_gates_list, num_2q_gates_list,
                     top_n=50, bottom_n=50, random_n=50, seed=42):
    """Get three groups of circuits: high-performance, low-performance, random"""
    random.seed(seed)
    np.random.seed(seed)

    # Create tuple list
    data_tuples = [(energy_list[i], i, circuit_list[i], depths[i], num_gates_list[i], num_2q_gates_list[i])
                  for i in range(len(energy_list))]

    # Sort by energy
    data_tuples.sort(key=lambda x: x[0])

    # Select top N (lowest energy) and bottom N (highest energy)
    top_circuits = data_tuples[:top_n]
    bottom_circuits = data_tuples[-bottom_n:][::-1]

    # Random select middle circuits (exclude top and bottom)
    available_indices = list(range(top_n, len(data_tuples) - bottom_n))
    random_indices = random.sample(available_indices, min(random_n, len(available_indices)))
    random_circuits = [(data_tuples[i][0], data_tuples[i][1], data_tuples[i][2],
                        data_tuples[i][3], data_tuples[i][4], data_tuples[i][5])
                       for i in random_indices]

    print(f"\nSelected {len(top_circuits)} top circuits (lowest energy):")
    print(f"Energy range: [{top_circuits[0][0]:.6f}, {top_circuits[-1][0]:.6f}]")

    print(f"\nSelected {len(bottom_circuits)} bottom circuits (highest energy):")
    print(f"Energy range: [{bottom_circuits[0][0]:.6f}, {bottom_circuits[-1][0]:.6f}]")

    print(f"\nSelected {len(random_circuits)} random circuits:")
    energies = [c[0] for c in random_circuits]
    print(f"Energy range: [{min(energies):.6f}, {max(energies):.6f}]")

    return top_circuits, bottom_circuits, random_circuits

def format_circuit_for_llm(circuit, circuit_id, energy, depth, num_gate, num_2q_gate):
    """Format a single circuit for LLM input"""
    # Use the passed parameters directly to avoid redundant computation
    # Only compute gate_counts which were not passed as parameters
    gate_counts = {'rx': 0, 'ry': 0, 'rz': 0, 'cx': 0, 'cz': 0}
    two_qubit_ratio = num_2q_gate / num_gate if num_gate > 0 else 0

    for gate in circuit:
        gate_type = gate[0]
        if gate_type in gate_counts:
            gate_counts[gate_type] += 1

    # Format gate sequence
    gate_sequence = []
    for i, gate in enumerate(circuit):
        gate_type = gate[0]
        qubits = gate[1:]
        gate_name = GATE_MAPPING.get(gate_type, gate_type)

        if len(qubits) == 2:
            gate_str = f"  Layer {i}: {gate_name} on qubits [{qubits[0]}, {qubits[1]}]"
        else:
            gate_str = f"  Layer {i}: {gate_name} on qubit [{qubits[0]}]"

        gate_sequence.append(gate_str)

    formatted = f"""
=== Circuit {circuit_id} ===
Energy: {energy:.8f}
Structural Info:
  - Circuit depth: {depth}
  - Total gates: {num_gate}
  - 2-qubit gates: {num_2q_gate}
  - Gate distribution: RX={gate_counts['rx']}, RY={gate_counts['ry']}, RZ={gate_counts['rz']}, CNOT={gate_counts['cx']}, CZ={gate_counts['cz']}
  - 2-qubit ratio: {two_qubit_ratio:.3f}

Gate sequence:
{chr(10).join(gate_sequence)}
"""
    return formatted

def build_llm_prompt(top_circuits, bottom_circuits, random_circuits, num_qubits):
    """Build LLM prompt for learning design principles using dataset-specific prompts"""
    # Get dataset-specific prompts
    if DATASET not in DATASET_PROMPTS:
        raise ValueError(f"Unknown DATASET: {DATASET}. Available: {list(DATASET_PROMPTS.keys())}")

    dataset_prompt = DATASET_PROMPTS[DATASET]

    # Format circuit information for three groups
    top_descriptions = []
    for energy, idx, circuit, depth, num_gate, num_2q_gate in top_circuits:
        circuit_desc = format_circuit_for_llm(circuit, idx, energy, depth, num_gate, num_2q_gate)
        top_descriptions.append(circuit_desc)
    top_text = '\n'.join(top_descriptions)

    bottom_descriptions = []
    for energy, idx, circuit, depth, num_gate, num_2q_gate in bottom_circuits:
        circuit_desc = format_circuit_for_llm(circuit, idx, energy, depth, num_gate, num_2q_gate)
        bottom_descriptions.append(circuit_desc)
    bottom_text = '\n'.join(bottom_descriptions)

    random_descriptions = []
    for energy, idx, circuit, depth, num_gate, num_2q_gate in random_circuits:
        circuit_desc = format_circuit_for_llm(circuit, idx, energy, depth, num_gate, num_2q_gate)
        random_descriptions.append(circuit_desc)
    random_text = '\n'.join(random_descriptions)

    # Use dataset-specific system prompt directly (no ground_state_energy placeholder)
    system_prompt = dataset_prompt["system_prompt"]

    # Format user prompt with circuit information
    user_prompt = dataset_prompt["user_prompt_intro"].format(
        num_top=len(top_circuits),
        num_bottom=len(bottom_circuits),
        num_random=len(random_circuits),
        top_text=top_text,
        bottom_text=bottom_text,
        random_text=random_text
    )

    return system_prompt, user_prompt

def call_llm(system_prompt, user_prompt):
    """Call LLM API to get design principles"""
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"API Base URL: {client.base_url}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print("\nCalling LLM to learn innovative design principles...")
    print("This may take a moment...")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            n=1
        )

        result = response.choices[0].message.content
        print("Successfully received response from LLM")
        return result

    except Exception as e:
        print(f"Error calling LLM API: {e}")
        raise

def save_principles(principles, output_path):
    """Save learned principles to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Quantum Circuit Innovative Design Principles v3 - Triple Comparison\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated by: {MODEL}\n")
        f.write("Task: VQE Quantum Circuit Architecture Design\n")
        f.write(f"Analysis method: Triple comparison (Top {TOP_N_CIRCUITS} vs Bottom {BOTTOM_N_CIRCUITS} vs Random {RANDOM_N_CIRCUITS})\n")
        f.write(f"Qubit count: {NUM_QUBITS}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(principles)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("Principles Analysis Complete\n")
        f.write("=" * 80 + "\n")

    print(f"Principles saved to: {output_path}")

def main():
    """Main function"""
    print("=" * 80)
    print("Quantum Circuit Design Principle Learning v3 - Triple Comparison")
    print(f"Dataset: {DATASET}")
    print("=" * 80)

    # Step 1: Load circuit data
    print("\n[Step 1] Loading circuit data...")
    circuit_list, energy_list, depths, num_gates_list, num_2q_gates_list = load_circuit_data(SAMPLES_PATH, ENERGY_PATH)

    # Step 2: Extract three groups of circuits
    print("\n[Step 2] Extracting three groups of circuits...")
    top_circuits, bottom_circuits, random_circuits = get_three_groups(
        circuit_list, energy_list, depths, num_gates_list, num_2q_gates_list,
        top_n=TOP_N_CIRCUITS,
        bottom_n=BOTTOM_N_CIRCUITS,
        random_n=RANDOM_N_CIRCUITS,
        seed=RANDOM_SEED
    )

    # Step 3: Build LLM prompt
    print("\n[Step 3] Building LLM prompt for triple comparison analysis...")
    system_prompt, user_prompt = build_llm_prompt(top_circuits, bottom_circuits, random_circuits, NUM_QUBITS)

    # Step 4: Call LLM to learn innovative principles
    print("\n[Step 4] Calling LLM to learn innovative design principles...")
    principles_response = call_llm(system_prompt, user_prompt)

    # Step 5: Save principles
    print("\n[Step 5] Saving learned principles...")
    save_principles(principles_response, OUTPUT_PATH)

    # Step 6: Show summary
    print("\n[Step 6] Principle Learning Complete!")
    print("=" * 80)
    print("\nInnovative Design Principles Summary:")
    print("-" * 40)

    # Print principles preview
    lines = principles_response.split('\n')
    for i, line in enumerate(lines[:100]):
        print(line)
    if len(lines) > 100:
        print(f"...\n[Full principles see {OUTPUT_PATH}]")

    print("\n" + "=" * 80)
    print(f"Full principles saved to: {OUTPUT_PATH}")

    return principles_response

if __name__ == "__main__":
    principles = main()
