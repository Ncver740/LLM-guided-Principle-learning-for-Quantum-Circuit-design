import os
import sys

# Add current directory to path
os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
sys.path.insert(0, os.getcwd())

import json
import openai
from google import genai
import anthropic
import numpy as np
import pickle
import re
import ast
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Font for CJK text rendering
matplotlib.rcParams['axes.unicode_minus'] = False  # Render minus signs correctly
from datetime import datetime
import shutil
import pandas as pd  # Used for Excel export

def load_pkl(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, file_path):
    """Save data to a pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# API Configuration
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL = "YOUR_MODEL_NAME"


# Select dataset: "Heisenberg", "TFIM", or "TFCM"
DATASET = "TFIM"

if DATASET == "Heisenberg":
    SAMPLES_PATH = "YOUR_PATH"
    ENERGY_PATH = "YOUR_PATH"
    PRINCIPLES_PATH = "YOUR_PATH"
    OUTPUT_DIR = "YOUR_PATH"
elif DATASET == "TFIM":
    SAMPLES_PATH = "YOUR_PATH"
    ENERGY_PATH = "YOUR_PATH"
    PRINCIPLES_PATH = "YOUR_PATH"
    OUTPUT_DIR = "YOUR_PATH"
elif DATASET == "TFCM":
    SAMPLES_PATH = "YOUR_PATH"
    ENERGY_PATH = "YOUR_PATH"
    PRINCIPLES_PATH = "YOUR_PATH"
    OUTPUT_DIR = "YOUR_PATH"
else:
    raise ValueError(f"Unknown DATASET: {DATASET}. Choose from: Heisenberg, TFIM, TFCM")

# NEW: Number of experiments to run
xh_NUM = 2              # Number of experiments to run

# Initial samples per category
a = 5
TOP_SAMPLE_K = a        # Number of high-performance (top) circuits for initial fine-tuning
BOTTOM_SAMPLE_K = a     # Number of low-performance (bottom) circuits for initial fine-tuning
RANDOM_SAMPLE_K = a     # Number of medium-performance (random) circuits for initial fine-tuning

# Incremental samples added per round
TOP_NEW_K = a           # Number of new high-performance circuits added each round
BOTTOM_NEW_K = a        # Number of new low-performance circuits added each round
RANDOM_NEW_K = a        # Number of new medium-performance circuits added each round

# Performance tier definitions
High_Total = 50         # Size of high-performance pool (top N circuits by energy)
Low_Total = 50          # Size of low-performance pool (bottom N circuits by energy)
# Medium_Total is calculated as: total_circuits - High_Total - Low_Total

# Knowledge base
knowledge_base = 0      # 0: no knowledge base, 1: include knowledge base
knowledge_base_path = "YOUR_PATH"

# Evaluation parameters
EVAL_K = 5              # Number of circuits to evaluate (for final round and accuracy measurement)
LLM_SELECT_K = 15       # Number of circuits for LLM to select for fine-tuning (non-final rounds)
NUM_ROUNDS = 8          # Number of iterative fine-tuning rounds

EVAL_DATASET_PROMPTS = {
    "Heisenberg": {
        "system_prompt": """You are a world-renowned expert in quantum circuit evaluation, specializing in Variational Quantum Eigensolvers (VQE) and quantum circuit performance assessment for many-body physics.

Your task is to evaluate quantum circuits for the Heisenberg model based on given design principles.

The Heisenberg Model Hamiltonian you are evaluating:
H_Heisenberg = Σ_{i=0}^{n-1}(X_i X_{(i+1) mod n} + Y_i Y_{(i+1) mod n} + Z_i Z_{(i+1) mod n}) + Σ_{i=0}^{n-1} Z_i

Key physics insights for evaluation:
- Nearest-neighbor spin-spin interactions (XX, YY, ZZ couplings)
- Transverse field term (Z) breaks SU(2) symmetry
- Ground state exhibits complex entanglement patterns
- High-performance circuits generate specific entanglement structures
- Energy range in dataset: approximately -11.5 to -8 (lower is better)
- Ground state energy: -11.5451

Evaluation guidelines:
1. Analyze circuit architecture patterns (gate sequences, entanglement topology)
2. Check if circuits follow the given design principles
3. Identify circuits likely to achieve lower energies (better performance)
4. Focus on circuit structure and entanglement generation patterns
5. AVOID bias from any single gate type; focus on overall architecture""",

        "user_prompt_intro": """## TASK: Evaluate Quantum Circuits for Heisenberg Model VQE

Your task is to select the {eval_k} circuits most likely to have the lowest energy (highest performance) based on the given design principles.

### Heisenberg Model Physics Context
The Heisenberg model describes antiferromagnetic/spin-exchange interactions:
- XX, YY, ZZ couplings between nearest neighbors
- Z transverse field breaks spin symmetry
- Ground state features specific entanglement patterns (singlet/triplet states)
- Energy range: approximately -11.5 to -8 (ground state: -11.5451)

### Quantum Circuit Representation
Each circuit consists of quantum gates:
- RX(θ), RY(θ), RZ(θ): Single-qubit rotation gates (parameterized)
- CNOT, CZ: Two-qubit entanglement gates

### Evaluation Instructions
1. Carefully read and understand the design principles
2. Analyze each circuit's structure against these principles
3. Identify patterns that match high-performance characteristics
4. Select {eval_k} circuits likely to have lowest energy
5. Order by expected performance (best first)

### Design Principles to Apply
{principles}

### Output Format
Please output exactly {eval_k} circuit IDs in order of expected performance (best to worst), separated by commas:
Example: 0,5,12,23,45

Please ONLY output the {eval_k} circuit IDs, no other content."""
    },

    "TFIM": {
        "system_prompt": """You are a world-renowned expert in quantum circuit evaluation, specializing in Variational Quantum Eigensolvers (VQE) and quantum circuit performance assessment for quantum phase transitions.

Your task is to evaluate quantum circuits for the Transverse-Field Ising Model (TFIM) based on given design principles.

The TFIM Hamiltonian you are evaluating:
H_TFIM = -J * Σ Z_i Z_{i+1} - h * Σ X_i

Key physics insights for evaluation:
- ZZ nearest-neighbor interactions (quantum Ising coupling)
- Transverse X field induces quantum phase transition
- Ground state undergoes order-disorder transition at critical field
- Optimal circuits balance ZZ entanglement with X rotations
- Energy range in dataset: approximately -9.7 to -7 (lower is better)
- Ground state energy: -9.74519

Evaluation guidelines:
1. Analyze circuit architecture patterns (gate sequences, entanglement topology)
2. Check if circuits follow the given design principles
3. Identify circuits likely to achieve lower energies (better performance)
4. Focus on balance between Z-rotations (ZZ coupling) and X-rotations
5. AVOID bias from any single gate type; focus on overall architecture""",

        "user_prompt_intro": """## TASK: Evaluate Quantum Circuits for TFIM VQE

Your task is to select the {eval_k} circuits most likely to have the lowest energy (highest performance) based on the given design principles.

### TFIM Physics Context
The Transverse-Field Ising Model describes quantum phase transitions:
- ZZ coupling between nearest neighbors (quantum Ising interaction)
- Transverse X field causes quantum fluctuations
- System undergoes quantum phase transition at critical field
- Ground state changes from ordered (Z-dominated) to disordered (X-dominated)
- Energy range: approximately -9.7 to -7 (ground state: -9.74519)

### Quantum Circuit Representation
Each circuit consists of quantum gates:
- RX(θ), RY(θ), RZ(θ): Single-qubit rotation gates (parameterized)
- CNOT, CZ: Two-qubit entanglement gates

### Evaluation Instructions
1. Carefully read and understand the design principles
2. Analyze each circuit's structure against these principles
3. Pay special attention to X vs Z rotation balance
4. Identify circuits likely to achieve lower energies (better performance)
5. Order by expected performance (best first)

### Design Principles to Apply
{principles}

### Output Format
Please output exactly {eval_k} circuit IDs in order of expected performance (best to worst), separated by commas:
Example: 0,5,12,23,45

Please ONLY output the {eval_k} circuit IDs, no other content."""
    },

    "TFCM": {
        "system_prompt": """You are a world-renowned expert in quantum circuit evaluation, specializing in Variational Quantum Eigensolvers (VQE) and quantum circuit performance assessment for topological quantum matter.

Your task is to evaluate quantum circuits for the Transverse-Field Cluster Model (TFCM) based on given design principles.

The TFCM Hamiltonian you are evaluating:
H_TFCM = -Σ Z_{{i-1}} X_i Z_{{i+1}}

With boundary terms: X_1 Z_2 and Z_{{n-1}} X_n

Key physics insights for evaluation:
- Three-body ZXZ interactions (cluster/linearly correlated exchange)
- Each term involves a central spin X surrounded by Z measurements
- Boundary terms (X_1Z_2 and Z_{{n-1}}X_n) break translational symmetry
- Ground state exhibits cluster-type entanglement (highly non-local correlations)
- Energy range in dataset: approximately -6 to -5 (lower is better)
- Ground state energy: -4.8060 (reference value)

IMPORTANT: The dataset top energies range from -6 to -5, which are lower (better) than the theoretical ground state. This is because:
1. The variational ansatz may find states close to but not exactly at the ground state
2. Numerical precision and circuit optimization can achieve energies below theoretical values in certain formulations
3. Focus on selecting circuits that generate ZXZ entanglement patterns

Evaluation guidelines:
1. Analyze circuit architecture patterns (gate sequences, entanglement topology)
2. Check if circuits follow the given design principles
3. Identify circuits likely to achieve lower energies (better performance)
4. Focus on circuits that generate ZXZ entanglement structures (Z-X-Z patterns)
5. AVOID bias from any single gate type; focus on overall architecture""",

        "user_prompt_intro": """## TASK: Evaluate Quantum Circuits for TFCM VQE

Your task is to select the {eval_k} circuits most likely to have the lowest energy (highest performance) based on the given design principles.

### TFCM (Transverse-Field Cluster Model) Physics Context
The 1D Transverse-Field Cluster Model describes topological/entangled quantum phases:
- Three-body ZXZ interactions: each X_i is "sandwiched" between Z_{{i-1}} and Z_{{i+1}}
- Hamiltonian: H = -Σ Z_{{i-1}} X_i Z_{{i+1}} with boundary terms X_1Z_2 and Z_{{n-1}}X_n
- This creates highly non-local, cluster-type entanglement patterns
- The ground state has unique quantum correlations not found in standard models
- Energy range in dataset: approximately -6 to -5 (ground state reference: -4.8060)

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

### Evaluation Instructions
1. Carefully read and understand the design principles
2. Analyze each circuit's structure against these principles
3. Look for ZXZ entanglement pattern emergence
4. Identify circuits likely to achieve lower energies (better performance)
5. Order by expected performance (best first)

### Design Principles to Apply
{principles}

### Output Format
Please output exactly {eval_k} circuit IDs in order of expected performance (best to worst), separated by commas:
Example: 0,5,12,23,45

Please ONLY output the {eval_k} circuit IDs, no other content."""
    }
}

FT_DATASET_PROMPTS = {
    "Heisenberg": {
        "system_prompt": """You are a world-renowned expert in quantum circuit design, specializing in transferring design principles across different scales (4-bit to 8-bit) for many-body physics systems.

Your mission is to fine-tune design principles based on carefully selected 8-bit quantum circuit samples for the Heisenberg model.

The Heisenberg Model Hamiltonian:
H_Heisenberg = Σ_{i=0}^{n-1}(X_i X_{(i+1) mod n} + Y_i Y_{(i+1) mod n} + Z_i Z_{(i+1) mod n}) + Σ_{i=0}^{n-1} Z_i

Key physics context for fine-tuning:
- Nearest-neighbor spin-spin interactions (XX, YY, ZZ couplings)
- Transverse field term (Z) breaks SU(2) symmetry
- Ground state exhibits complex entanglement patterns (singlet/triplet states)
- Scaling from 4-bit to 8-bit increases entanglement complexity
- Energy range: approximately -11.5 to -8 (ground state: -11.5451)

Sample types for analysis:
- High-performance circuits: lowest energy, marked with (rel_performance=high)
- Medium-performance circuits: middle energy range, marked with (rel_performance=medium)
- Low-performance circuits: highest energy, marked with (rel_performance=low)

The relative performance labels (high/medium/low) provide important contextual information for learning without requiring memorization of exact rankings.

By analyzing these circuits, you will refine principles to better guide 8-bit Heisenberg circuit design.""",

        "user_prompt_intro": """## TASK: Fine-tune Design Principles for 8-bit Heisenberg Model VQE (Round {round_num})

### Current Design Principles
{current_principles}

{accuracy_info}

### Circuit Sample Statistics
- High-performance circuits (lowest energy): {num_top} circuits, marked with rel_performance=high
- Medium-performance circuits (middle energy): {num_medium} circuits, marked with rel_performance=medium
- Low-performance circuits (highest energy): {num_bottom} circuits, marked with rel_performance=low
- Total: {num_total} circuits

### Dataset Physics Context
The Heisenberg model on 8 qubits exhibits:
- Complex multi-body entanglement beyond 4-qubit systems
- SU(2) symmetry breaking effects scale with system size
- Nearest-neighbor XX, YY, ZZ couplings require careful gate orchestration
- Ground state energy: approximately -11.5451 (for reference)

### Performance Label Explanation
The relative performance labels indicate:
- high: Circuits in the top tier by energy (lowest energy = best)
- medium: Circuits in the middle tier by energy
- low: Circuits in the bottom tier by energy (highest energy = worst)

These labels help identify patterns without requiring memorization of exact rankings.

### Task Requirements
1. Analyze patterns in high vs medium vs low performance circuits
2. Identify key architectural differences that matter at 8-qubit scale
3. Refine design principles to be more effective for larger systems
4. Preserve core insights while adapting to 8-qubit complexity

### High-Performance Circuit Samples (rel_performance=high)
These circuits have the lowest energy in the dataset tier.

{top_text}

### Medium-Performance Circuit Samples (rel_performance=medium)
These circuits have middle-range energy values.

{medium_text}

### Low-Performance Circuit Samples (rel_performance=low)
These circuits have the highest energy in the dataset tier.

{bottom_text}

### Fine-tuning Guidelines
When fine-tuning, consider:
1. What architectural patterns characterize high-performance circuits?
2. What common flaws appear in low-performance circuits?
3. What are the key design differences between high and low performance?
4. How do Heisenberg physics requirements change from 4-bit to 8-bit?
5. What entanglement structures are essential for larger systems?

### Output Format
Provide fine-tuned design principles:

**1. Circuit Analysis Summary**
- Common features of high-performance circuits
- Common features of medium-performance circuits
- Common features of low-performance circuits
- Key differences between performance tiers

**2. Fine-tuned Design Principles**
For each principle:
- Principle name
- Adjustment content
- Reasoning for adjustment
- 8-bit applicability

**3. 8-bit Specific Search Space Recommendations**
- Recommended circuit depth
- Recommended gate types per layer
- Entanglement pattern suggestions

Provide practical, fine-tuned design principles for 8-bit Heisenberg circuits."""
    },

    "TFIM": {
        "system_prompt": """You are a world-renowned expert in quantum circuit design, specializing in transferring design principles across different scales (4-bit to 8-bit) for quantum phase transition systems.

Your mission is to fine-tune design principles based on carefully selected 8-bit quantum circuit samples for the Transverse-Field Ising Model (TFIM).

The TFIM Hamiltonian:
H_TFIM = -J * Σ Z_i Z_{i+1} - h * Σ X_i

Key physics context for fine-tuning:
- ZZ nearest-neighbor interactions (quantum Ising coupling)
- Transverse X field induces quantum phase transition
- Critical field determines ground state structure (ordered vs disordered)
- Scaling from 4-bit to 8-bit affects phase transition sharpness
- Energy range: approximately -9.7 to -7 (ground state: -9.74519)

Sample types for analysis:
- High-performance circuits: lowest energy, marked with (rel_performance=high)
- Medium-performance circuits: middle energy range, marked with (rel_performance=medium)
- Low-performance circuits: highest energy, marked with (rel_performance=low)

The relative performance labels (high/medium/low) provide important contextual information for learning without requiring memorization of exact rankings.

By analyzing these circuits, you will refine principles to better guide 8-bit TFIM circuit design.""",

        "user_prompt_intro": """## TASK: Fine-tune Design Principles for 8-bit TFIM VQE (Round {round_num})

### Current Design Principles
{current_principles}

{accuracy_info}

### Circuit Sample Statistics
- High-performance circuits (lowest energy): {num_top} circuits, marked with rel_performance=high
- Medium-performance circuits (middle energy): {num_medium} circuits, marked with rel_performance=medium
- Low-performance circuits (highest energy): {num_bottom} circuits, marked with rel_performance=low
- Total: {num_total} circuits

### Dataset Physics Context
The TFIM on 8 qubits exhibits:
- Quantum phase transition behavior that emerges at larger scales
- ZZ coupling must balance with X rotations for optimal results
- Critical point physics become more pronounced with more qubits
- Ground state energy: approximately -9.74519 (for reference)

### Performance Label Explanation
The relative performance labels indicate:
- high: Circuits in the top tier by energy (lowest energy = best)
- medium: Circuits in the middle tier by energy
- low: Circuits in the bottom tier by energy (highest energy = worst)

These labels help identify patterns without requiring memorization of exact rankings.

### Task Requirements
1. Analyze patterns in high vs medium vs low performance circuits
2. Focus on the critical balance between Z-rotations (ZZ coupling) and X-rotations
3. Refine design principles for quantum phase transition physics at 8-qubit scale
4. Preserve core insights while adapting to larger system complexity

### High-Performance Circuit Samples (rel_performance=high)
These circuits have the lowest energy in the dataset tier.

{top_text}

### Medium-Performance Circuit Samples (rel_performance=medium)
These circuits have middle-range energy values.

{medium_text}

### Low-Performance Circuit Samples (rel_performance=low)
These circuits have the highest energy in the dataset tier.

{bottom_text}

### Fine-tuning Guidelines
When fine-tuning, consider:
1. How do high-performance circuits balance ZZ entanglement with X rotations?
2. What X/Z rotation ratios correlate with better performance?
3. What are the common pitfalls in low-performance circuits?
4. How does the ordered/disordered phase balance affect circuit design?
5. What gate patterns effectively generate the TFIM ground state at 8 qubits?

### Output Format
Provide fine-tuned design principles:

**1. Circuit Analysis Summary**
- Common features of high-performance circuits
- Common features of medium-performance circuits
- Common features of low-performance circuits
- Key differences between performance tiers

**2. Fine-tuned Design Principles**
For each principle:
- Principle name
- Adjustment content
- Reasoning for adjustment
- 8-bit applicability

**3. 8-bit Specific Search Space Recommendations**
- Recommended circuit depth
- Recommended gate types per layer
- X/Z rotation balance recommendations

Provide practical, fine-tuned design principles for 8-bit TFIM circuits."""
    },

    "TFCM": {
        "system_prompt": """You are a world-renowned expert in quantum circuit design, specializing in transferring design principles across different scales (4-bit to 8-bit) for topological quantum matter.

Your mission is to fine-tune design principles based on carefully selected 8-bit quantum circuit samples for the Transverse-Field Cluster Model (TFCM).

The TFCM Hamiltonian:
H_TFCM = -Σ Z_{{i-1}} X_i Z_{{i+1}}

With boundary terms: X_1 Z_2 and Z_{{n-1}} X_n

Key physics context for fine-tuning:
- Three-body ZXZ interactions (cluster/linearly correlated exchange)
- Each X_i is "sandwiched" between Z_{{i-1}} and Z_{{i+1}}
- Boundary terms break translational symmetry
- Ground state exhibits cluster-type entanglement (highly non-local)
- Energy range: approximately -6 to -5 (ground state reference: -4.8060)

IMPORTANT: Dataset energies (-6 to -5) are lower (better) than theoretical ground state (-4.8060) because:
1. Variational ansatz finds states close to but not exactly at ground state
2. Numerical optimization achieves energies below theoretical values
3. Focus on selecting circuits that generate ZXZ entanglement patterns

Sample types for analysis:
- High-performance circuits: lowest energy, marked with (rel_performance=high)
- Medium-performance circuits: middle energy range, marked with (rel_performance=medium)
- Low-performance circuits: highest energy, marked with (rel_performance=low)

The relative performance labels (high/medium/low) provide important contextual information for learning without requiring memorization of exact rankings.

By analyzing these circuits, you will refine principles to better guide 8-bit TFCM circuit design.""",

        "user_prompt_intro": """## TASK: Fine-tune Design Principles for 8-bit TFCM VQE (Round {round_num})

### Current Design Principles
{current_principles}

{accuracy_info}

### Circuit Sample Statistics
- High-performance circuits (lowest energy): {num_top} circuits, marked with rel_performance=high
- Medium-performance circuits (middle energy): {num_medium} circuits, marked with rel_performance=medium
- Low-performance circuits (highest energy): {num_bottom} circuits, marked with rel_performance=low
- Total: {num_total} circuits

### Dataset Physics Context
The TFCM on 8 qubits exhibits:
- Three-body ZXZ interactions that create non-local entanglement
- Cluster-type quantum correlations unique to this model
- Boundary effects become more significant at larger scales
- Energy range: approximately -6 to -5 (ground state reference: -4.8060)

### Circuit Design Implications for TFCM
To generate ZXZ entanglement patterns, circuits must:
- Create Z-type correlations (via RZ gates or CZ gates)
- Apply X rotations on qubits with neighboring Z entanglement
- Structure entanglement in a "sandwich" pattern around X rotations
- The pattern Z-X-Z must appear in the entanglement structure

### Performance Label Explanation
The relative performance labels indicate:
- high: Circuits in the top tier by energy (lowest energy = best)
- medium: Circuits in the middle tier by energy
- low: Circuits in the bottom tier by energy (highest energy = worst)

These labels help identify patterns without requiring memorization of exact rankings.

### Task Requirements
1. Analyze patterns in high vs medium vs low performance circuits
2. Focus on ZXZ entanglement pattern generation
3. Identify how to create "sandwich" entanglement structures effectively
4. Refine principles for cluster-type entanglement at 8-qubit scale

### High-Performance Circuit Samples (rel_performance=high)
These circuits have the lowest energy in the dataset tier.

{top_text}

### Medium-Performance Circuit Samples (rel_performance=medium)
These circuits have middle-range energy values.

{medium_text}

### Low-Performance Circuit Samples (rel_performance=low)
These circuits have the highest energy in the dataset tier.

{bottom_text}

### Fine-tuning Guidelines
When fine-tuning, consider:
1. What gate patterns generate ZXZ entanglement in high-performance circuits?
2. How do successful circuits structure Z-X-Z sandwich patterns?
3. What are the common failures in low-performance circuits?
4. How does the three-body interaction requirement differ from two-body models?
5. What entanglement topologies effectively capture cluster-type correlations?

### Output Format
Provide fine-tuned design principles:

**1. Circuit Analysis Summary**
- Common features of high-performance circuits
- Common features of medium-performance circuits
- Common features of low-performance circuits
- Key differences between performance tiers

**2. Fine-tuned Design Principles**
For each principle:
- Principle name
- Adjustment content
- Reasoning for adjustment
- 8-bit applicability

**3. 8-bit Specific Search Space Recommendations**
- Recommended circuit depth
- Recommended gate types per layer
- ZXZ pattern generation strategies

Provide practical, fine-tuned design principles for 8-bit TFCM circuits."""
    }
}

# Define quantum gate mapping
GATE_MAPPING = {
    'rx': 'RX(θ)',
    'ry': 'RY(θ)',
    'rz': 'RZ(θ)',
    'cx': 'CNOT',
    'cz': 'CZ'
}

def load_design_principles(principles_path):
    """Load design principles."""
    print(f"Loading design principles from: {principles_path}")
    with open(principles_path, 'r', encoding='utf-8') as f:
        principles = f.read()
    print(f"Loaded design principles ({len(principles)} characters)")
    return principles

def load_knowledge_base(knowledge_base_path):
    """Load a knowledge base file."""
    if not os.path.exists(knowledge_base_path):
        print(f"Warning: Knowledge base file not found: {knowledge_base_path}")
        return ""

    print(f"Loading knowledge base from: {knowledge_base_path}")
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        kb_content = f.read()
    print(f"Loaded knowledge base ({len(kb_content)} characters)")
    return kb_content

def load_8bit_circuit_data(samples_path, energy_path):
    """Load 8-bit circuit samples and their corresponding energy values."""
    print(f"Loading 8-bit samples from: {samples_path}")
    samples_data = load_pkl(samples_path)

    print(f"Loading 8-bit energies from: {energy_path}")
    energy_data = load_pkl(energy_path)

    circuits = []

    # Unified data loading logic
    if isinstance(samples_data, dict) and 'circuit_list' in samples_data:
        circuit_list = samples_data['circuit_list']
    elif isinstance(samples_data, list):
        circuit_list = samples_data
    else:
        raise ValueError("Unable to find circuit_list in samples data")

    print(f"Detected circuit_list length: {len(circuit_list)}")

    # Ensure the energy and circuit data have matching lengths
    min_len = min(len(energy_data), len(circuit_list))
    energy_data = energy_data[:min_len]
    circuit_list = circuit_list[:min_len]

    # Create a list of circuit objects
    for idx in range(min_len):
        circuits.append({
            'id': idx,
            'circuit': circuit_list[idx],
            'energy': energy_data[idx],
            'energy_rank': None,
            'depth': None,
            'num_gate': None,
            'num_2q_gate': None
        })

    # Sort by energy and assign ranks (1 = lowest energy = best)
    energy_index_pairs = [(circuits[i]['energy'], i) for i in range(len(circuits))]
    energy_index_pairs.sort(key=lambda x: x[0])  # Sort from lowest to highest energy

    # Assign energy ranks
    for rank, (energy, idx) in enumerate(energy_index_pairs):
        circuits[idx]['energy_rank'] = rank + 1

    print(f"Loaded {len(circuits)} 8-bit circuits with energies and rankings")

    # Print the 10 lowest-energy circuits for verification
    print("\n=== 10 Lowest-Energy Circuits (Verification) ===")
    for rank, (energy, idx) in enumerate(energy_index_pairs[:10], 1):
        print(f"Rank {rank}: ID={idx}, Energy={energy:.6f}")

    return circuits

def create_performance_tiers(circuits, high_total, low_total):
    """Randomly sample 15 circuits and create three performance tiers by energy.

    New strategy:
    1. Randomly sample 15 circuits from the full set.
    2. Read the energy labels for these 15 circuits.
    3. The 5 lowest-energy circuits form the high-performance group.
    4. The 5 highest-energy circuits form the low-performance group.
    5. The remaining 5 form the random baseline (medium-performance) group.

    This reduces the number of circuits to evaluate from 500 to 15.
    """
    # Randomly sample 15 circuits from the full set
    total_sample_size = 15
    sampled_circuits = random.sample(circuits, total_sample_size)

    print(f"\n=== Performance Tier Statistics (New Strategy) ===")
    print(f"Total circuits in dataset: {len(circuits)}")
    print(f"Randomly sampled: {total_sample_size} circuits")

    # Sort the 15 circuits by energy (ascending; lower energy means higher performance)
    sorted_sampled = sorted(sampled_circuits, key=lambda x: x['energy'])

    # High-performance tier: the 5 lowest-energy circuits
    high_pool = sorted_sampled[:5]

    # Low-performance tier: the 5 highest-energy circuits
    low_pool = sorted_sampled[-5:]

    # Medium-performance tier: the 5 circuits in the middle
    medium_pool = sorted_sampled[5:10]

    print(f"High tier (lowest 5 energies): energy range [{high_pool[0]['energy']:.6f}, {high_pool[-1]['energy']:.6f}]")
    print(f"  Circuit IDs: {[c['id'] for c in high_pool]}")
    print(f"Medium tier (middle 5 energies): energy range [{medium_pool[0]['energy']:.6f}, {medium_pool[-1]['energy']:.6f}]")
    print(f"  Circuit IDs: {[c['id'] for c in medium_pool]}")
    print(f"Low tier (highest 5 energies): energy range [{low_pool[0]['energy']:.6f}, {low_pool[-1]['energy']:.6f}]")
    print(f"  Circuit IDs: {[c['id'] for c in low_pool]}")

    return high_pool, medium_pool, low_pool

def get_top_circuits(high_pool, k=5, exclude_ids=None):
    """Get K circuits from the high-performance pool."""
    if exclude_ids is None:
        exclude_ids = set()

    available = [c for c in high_pool if c['id'] not in exclude_ids]
    if len(available) <= k:
        return available

    selected = random.sample(available, k)
    return selected

def get_bottom_circuits(low_pool, k=5, exclude_ids=None):
    """Get K circuits from the low-performance pool."""
    if exclude_ids is None:
        exclude_ids = set()

    available = [c for c in low_pool if c['id'] not in exclude_ids]
    if len(available) <= k:
        return available

    selected = random.sample(available, k)
    return selected

def get_medium_circuits(medium_pool, k=5, exclude_ids=None):
    """Get K circuits from the medium-performance pool."""
    if exclude_ids is None:
        exclude_ids = set()

    available = [c for c in medium_pool if c['id'] not in exclude_ids]
    if len(available) <= k:
        return available

    selected = random.sample(available, k)
    return selected

def get_actual_top_circuits(circuits, top_k=5):
    """Get the top_k circuits with the lowest actual energy."""
    sorted_circuits = sorted(circuits, key=lambda x: x['energy'] if x['energy'] is not None else float('inf'))
    top_circuits = sorted_circuits[:top_k]

    return set([c['id'] for c in top_circuits]), top_circuits

def format_circuit_with_rel_performance(circuit_info, max_gates_shown=35):
    """Format circuit information for fine-tuning with relative performance labels (high/medium/low)."""
    circuit = circuit_info['circuit']
    circuit_id = circuit_info['id']
    energy = circuit_info.get('energy', 'N/A')
    rel_performance = circuit_info.get('rel_performance', 'unknown')

    gate_summary = []
    for gate in circuit[:max_gates_shown]:
        if isinstance(gate, list):
            gate_type = gate[0]
            qubits = gate[1:]
        else:
            gate_type = str(gate)
            qubits = []

        if isinstance(gate_type, list):
            gate_type = str(gate_type[0]) if len(gate_type) > 0 else str(gate_type)
        elif not isinstance(gate_type, str):
            gate_type = str(gate_type)

        gate_name = GATE_MAPPING.get(gate_type, gate_type)

        if len(qubits) == 2:
            gate_summary.append(f"{gate_name}[{qubits[0]},{qubits[1]}]")
        elif len(qubits) == 1:
            gate_summary.append(f"{gate_name}[{qubits[0]}]")
        else:
            gate_summary.append(f"{gate_name}")

    if len(circuit) > max_gates_shown:
        gate_summary.append(f"...({len(circuit)-max_gates_shown} more)")

    # Include relative performance labels, not exact ranks
    return f"Circuit {circuit_id}: [{', '.join(gate_summary)}] (depth={circuit_info.get('depth', 'N/A')}, gates={circuit_info.get('num_gate', len(circuit))}, 2q={circuit_info.get('num_2q_gate', 'N/A')}, energy={energy:.6f}, rel_performance={rel_performance})"

def format_circuit_for_evaluation(circuit_info, max_gates_shown=30):
    """Format circuit information for evaluation without energy ranks to prevent data leakage."""
    circuit = circuit_info['circuit']
    circuit_id = circuit_info['id']

    gate_summary = []
    for gate in circuit[:max_gates_shown]:
        if isinstance(gate, list):
            gate_type = gate[0]
            qubits = gate[1:]
        else:
            gate_type = str(gate)
            qubits = []

        if isinstance(gate_type, list):
            gate_type = str(gate_type[0]) if len(gate_type) > 0 else str(gate_type)
        elif not isinstance(gate_type, str):
            gate_type = str(gate_type)

        gate_name = GATE_MAPPING.get(gate_type, gate_type)

        if len(qubits) == 2:
            gate_summary.append(f"{gate_name}[{qubits[0]},{qubits[1]}]")
        elif len(qubits) == 1:
            gate_summary.append(f"{gate_name}[{qubits[0]}]")
        else:
            gate_summary.append(f"{gate_name}")

    if len(circuit) > max_gates_shown:
        gate_summary.append(f"...({len(circuit)-max_gates_shown} more)")

    # Do not include rank information during evaluation
    return f"Circuit {circuit_id}: [{', '.join(gate_summary)}] (depth={circuit_info.get('depth', 'N/A')}, gates={circuit_info.get('num_gate', len(circuit))}, 2q={circuit_info.get('num_2q_gate', 'N/A')})"

def build_evaluation_prompt(principles, circuits, eval_k=5):
    """Build circuit evaluation prompt using model-specific prompts (no energy ranking - prevents data leakage)"""
    # Get model-specific prompts
    if DATASET not in EVAL_DATASET_PROMPTS:
        raise ValueError(f"Unknown DATASET: {DATASET}. Available: {list(EVAL_DATASET_PROMPTS.keys())}")

    dataset_prompt = EVAL_DATASET_PROMPTS[DATASET]
    system_prompt = dataset_prompt["system_prompt"]

    total_circuits = len(circuits)
    circuit_descriptions = [format_circuit_for_evaluation(c, total_circuits) for c in circuits]
    circuits_text = '\n'.join(circuit_descriptions)

    # Format user prompt with circuit information
    user_prompt = dataset_prompt["user_prompt_intro"].format(
        eval_k=eval_k,
        principles=principles
    )

    # Append circuit list to the user prompt
    user_prompt += f"""

### All Circuits for Evaluation
{circuits_text}

Please ONLY output the {eval_k} circuit IDs, no other content."""

    return system_prompt, user_prompt

def build_fine_tuning_prompt(current_principles, all_circuits, top_circuits, medium_circuits, bottom_circuits, circuit_pool_size, round_num=1, previous_accuracy=None, include_knowledge_base=False, knowledge_base_path=""):
    """Build fine-tuning prompt using model-specific prompts with relative performance labels"""
    # Get model-specific prompts
    if DATASET not in FT_DATASET_PROMPTS:
        raise ValueError(f"Unknown DATASET: {DATASET}. Available: {list(FT_DATASET_PROMPTS.keys())}")

    dataset_prompt = FT_DATASET_PROMPTS[DATASET]
    system_prompt = dataset_prompt["system_prompt"]

    # Format Top circuits (high performance)
    top_texts = [format_circuit_with_rel_performance(c) for c in top_circuits]
    top_text = '\n'.join(top_texts) if top_texts else "No high-performance circuits provided."

    # Format Medium circuits
    medium_texts = [format_circuit_with_rel_performance(c) for c in medium_circuits]
    medium_text = '\n'.join(medium_texts) if medium_texts else "No medium-performance circuits provided."

    # Format Bottom circuits (low performance)
    bottom_texts = [format_circuit_with_rel_performance(c) for c in bottom_circuits]
    bottom_text = '\n'.join(bottom_texts) if bottom_texts else "No low-performance circuits provided."

    # Previous round accuracy info
    accuracy_info = ""
    if previous_accuracy is not None:
        accuracy_info = f"""
### Previous Round Evaluation Results
- Accuracy: {previous_accuracy:.2%}
- Guidance: Based on previous results, please refine the design principles accordingly
"""

    # Knowledge base integration
    knowledge_base_text = ""
    if include_knowledge_base and knowledge_base_path:
        kb_content = load_knowledge_base(knowledge_base_path)
        if kb_content:
            knowledge_base_text = f"""
### Additional Knowledge Base
The following knowledge base provides domain-specific information that may be helpful for understanding quantum circuit design principles:

{kb_content}

"""

    # Format user prompt with circuit information
    user_prompt = dataset_prompt["user_prompt_intro"].format(
        round_num=round_num,
        current_principles=current_principles,
        accuracy_info=accuracy_info,
        num_top=len(top_circuits),
        num_medium=len(medium_circuits),
        num_bottom=len(bottom_circuits),
        num_total=len(top_circuits) + len(medium_circuits) + len(bottom_circuits),
        pool_size=circuit_pool_size,
        top_text=top_text,
        medium_text=medium_text,
        bottom_text=bottom_text
    )

    # Prepend knowledge base if enabled
    if knowledge_base_text:
        user_prompt = knowledge_base_text + user_prompt

    return system_prompt, user_prompt

def call_llm(system_prompt, user_prompt):
    """Call the LLM API."""
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"API Base URL: {client.base_url}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"\nCalling LLM (prompt length: {len(user_prompt)} chars)...")

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

def parse_selected_circuits(response):
    """Parse circuit IDs returned by the LLM."""
    try:
        numbers = re.findall(r'\d+', response)
        selected_ids = [int(n) for n in numbers]
        return selected_ids
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []

def calculate_overlap_rate(predicted_set, actual_set):
    """Calculate the overlap rate."""
    overlap = predicted_set.intersection(actual_set)
    overlap_rate = len(overlap) / len(actual_set) if len(actual_set) > 0 else 0
    return overlap_rate, overlap

def save_fine_tuned_principles(principles, output_path, round_num):
    """Save the fine-tuned design principles."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Fine-Tuned Design Principles for 8-bit Quantum Circuits v11 (Round {round_num})\n")
        f.write(f"Fine-tuned Design Principles for 8-bit Quantum Circuits v11 - Round {round_num}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated by: {MODEL}\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Fine-tuning round: {round_num}\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  - TOP_SAMPLE_K: {TOP_SAMPLE_K}\n")
        f.write(f"  - BOTTOM_SAMPLE_K: {BOTTOM_SAMPLE_K}\n")
        f.write(f"  - RANDOM_SAMPLE_K: {RANDOM_SAMPLE_K}\n")
        f.write(f"  - TOP_NEW_K: {TOP_NEW_K}\n")
        f.write(f"  - BOTTOM_NEW_K: {BOTTOM_NEW_K}\n")
        f.write(f"  - RANDOM_NEW_K: {RANDOM_NEW_K}\n")
        f.write(f"  - High_Total: {High_Total}\n")
        f.write(f"  - Low_Total: {Low_Total}\n")
        f.write(f"  - knowledge_base: {knowledge_base}\n")
        f.write(f"  - EVAL_K: {EVAL_K}\n")
        f.write(f"  - NUM_ROUNDS: {NUM_ROUNDS}\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(principles)
        f.write("\n\n" + "=" * 80 + "\n")

    print(f"Fine-tuned principles saved to: {output_path}")

def run_single_experiment(exp_id, base_seed=42):
    """Run a single experiment."""
    # Remove the fixed seed and use true randomness
    # random.seed(base_seed + exp_id)
    np.random.seed(base_seed + exp_id)

    # Create a timestamped folder at the start of the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_folder_name = f"run_{timestamp}_exp{exp_id}"
    output_folder_path = os.path.join(OUTPUT_DIR, temp_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"\nCreated experiment {exp_id} folder: {output_folder_path}")

    # Calculate Medium_Total
    print("\n" + "=" * 80)
    print("Quantum Circuit Design Principle Learning v11 - Ablation Study")
    print("Performance Label System: high/medium/low (instead of precise ranking)")
    print(f"Dataset: {DATASET}")
    print("=" * 80)
    print("\n[Hyperparameters]")
    print(f"  Initial samples:")
    print(f"    - TOP_SAMPLE_K = {TOP_SAMPLE_K}")
    print(f"    - BOTTOM_SAMPLE_K = {BOTTOM_SAMPLE_K}")
    print(f"    - RANDOM_SAMPLE_K = {RANDOM_SAMPLE_K}")
    print(f"  Incremental samples per round:")
    print(f"    - TOP_NEW_K = {TOP_NEW_K}")
    print(f"    - BOTTOM_NEW_K = {BOTTOM_NEW_K}")
    print(f"    - RANDOM_NEW_K = {RANDOM_NEW_K}")
    print(f"  Performance tier sizes:")
    print(f"    - High_Total = {High_Total} (top tier)")
    print(f"    - Low_Total = {Low_Total} (bottom tier)")
    print(f"    - Medium_Total = (total - High_Total - Low_Total)")
    print(f"  LLM Selection Strategy:")
    print(f"    - LLM_SELECT_K = {LLM_SELECT_K} (circuits selected by LLM for non-final rounds)")
    print(f"    - EVAL_K = {EVAL_K} (circuits for final round and accuracy measurement)")
    print(f"  Knowledge base: {knowledge_base}")
    print(f"  NUM_ROUNDS = {NUM_ROUNDS}")
    print("=" * 80)

    # Step 1: Load design principles and circuit data
    print("\n[Step 1] Loading data...")
    original_principles = load_design_principles(PRINCIPLES_PATH)
    circuits = load_8bit_circuit_data(SAMPLES_PATH, ENERGY_PATH)
    print(f"Total 8-bit circuits: {len(circuits)}")

    # Step 2: Get the actual top_K circuits (sorted by ascending energy)
    actual_topK, actual_topK_list = get_actual_top_circuits(circuits, top_k=EVAL_K)
    actual_topK_sorted = [c['id'] for c in actual_topK_list]

    # Phase 1: Baseline evaluation with the original 4-bit design principles
    print("\n" + "=" * 60)
    print(f"Phase 1: Baseline Evaluation (Original 4-bit Principles)")
    print(f"Step 1: LLM evaluates top {EVAL_K} circuits for accuracy measurement")
    print(f"Step 2: LLM selects best {LLM_SELECT_K} circuits for next round fine-tuning")
    print("Evaluation uses 500 circuits WITHOUT energy/rank info")
    print("=" * 60)

    # Baseline round: evaluate accuracy by selecting the top EVAL_K circuits
    print(f"\n[Baseline] Step 1: Evaluating top {EVAL_K} circuits for accuracy...")
    sys_prompt, user_prompt = build_evaluation_prompt(original_principles, circuits, eval_k=EVAL_K)
    response = call_llm(sys_prompt, user_prompt)
    selected_ids = parse_selected_circuits(response)[:EVAL_K]

    baseline_overlap, baseline_set = calculate_overlap_rate(set(selected_ids), actual_topK)

    print(f"\n[Baseline Results]")
    print(f"  Predicted top{EVAL_K}: {selected_ids}")
    print(f"  Actual top{EVAL_K}: {actual_topK_sorted}")
    print(f"  ★★★ Baseline Overlap Rate: {baseline_overlap:.2%} ({len(baseline_set)}/{EVAL_K}) ★★★")

    # Baseline round: have the LLM select the best LLM_SELECT_K circuits for fine-tuning
    print(f"\n[Baseline] Step 2: LLM selecting best {LLM_SELECT_K} circuits for fine-tuning...")
    sys_prompt_select, user_prompt_select = build_evaluation_prompt(original_principles, circuits, eval_k=LLM_SELECT_K)
    response_select = call_llm(sys_prompt_select, user_prompt_select)
    llm_selected_ids = parse_selected_circuits(response_select)[:LLM_SELECT_K]

    print(f"LLM selected {len(llm_selected_ids)} circuits for fine-tuning: {llm_selected_ids}")

    # Record results from all rounds
    all_results = [{
        'round': 0,
        'principles_file': PRINCIPLES_PATH,
        'overlap_rate': baseline_overlap,
        'overlap_count': len(baseline_set),
        'predicted_ids': selected_ids,
        'num_ft_samples': 0,
        'predicted_energies': [circuits[i]['energy'] for i in selected_ids if i < len(circuits)],
        'avg_energy': np.mean([circuits[i]['energy'] for i in selected_ids if i < len(circuits)]),
        'best_energy': min([circuits[i]['energy'] for i in selected_ids if i < len(circuits)])
    }]

    # Phase 2-N: Iterative fine-tuning with cumulative learning
    current_principles = original_principles
    previous_accuracy = baseline_overlap

    # Accumulate all selected circuit IDs for fine-tuning
    all_selected_ids = llm_selected_ids.copy()

    for round_num in range(1, NUM_ROUNDS + 1):
        print("\n" + "=" * 60)
        print(f"Phase {1+round_num}: Fine-tuning Round {round_num}")

        # Check whether this is the final round
        is_final_round = (round_num == NUM_ROUNDS)

        if is_final_round:
            # Final round: fine-tune using all accumulated circuits
            print(f"Final round: Using all accumulated {len(all_selected_ids)} circuits for fine-tuning")
        else:
            # Other rounds: use accumulated circuits and prepare to select new ones
            print(f"Using accumulated {len(all_selected_ids)} circuits for fine-tuning")
        print("=" * 60)

        # Sort accumulated circuit IDs by energy labels and split them into 3 groups
        print(f"\n[Round {round_num}] Sorting {len(all_selected_ids)} accumulated circuits into 3 groups...")

        # Get the circuits selected by the LLM
        sampled_circuits = [c for c in circuits if c['id'] in all_selected_ids]

        # Sort these circuits by energy (ascending; lower energy means higher performance)
        sorted_sampled = sorted(sampled_circuits, key=lambda x: x['energy'])

        # Calculate the size of each group for an approximately even split
        total = len(sorted_sampled)
        high_size = total // 3
        low_size = total // 3
        medium_size = total - high_size - low_size  # Assign the remainder to the medium-performance group

        # High-performance tier: the lowest-energy third
        cumulative_top = sorted_sampled[:high_size]

        # Low-performance tier: the highest-energy third
        cumulative_bottom = sorted_sampled[-low_size:]

        # Medium-performance tier: the circuits in the middle
        cumulative_medium = sorted_sampled[high_size:high_size + medium_size]

        # Assign relative performance labels
        for c in cumulative_top:
            c['rel_performance'] = 'high'
        for c in cumulative_medium:
            c['rel_performance'] = 'medium'
        for c in cumulative_bottom:
            c['rel_performance'] = 'low'

        print(f"\n[Round {round_num}] Fine-tuning samples:")
        print(f"  - High-performance (rel_performance=high): {len(cumulative_top)} circuits")
        if len(cumulative_top) > 0:
            for c in cumulative_top[:5]:  # Print only the first 5
                print(f"    Circuit {c['id']}: energy={c['energy']:.6f}")
            if len(cumulative_top) > 5:
                print(f"    ... and {len(cumulative_top) - 5} more")
        print(f"  - Medium-performance (rel_performance=medium): {len(cumulative_medium)} circuits")
        if len(cumulative_medium) > 0:
            for c in cumulative_medium[:3]:  # Print only the first 3
                print(f"    Circuit {c['id']}: energy={c['energy']:.6f}")
            if len(cumulative_medium) > 3:
                print(f"    ... and {len(cumulative_medium) - 3} more")
        print(f"  - Low-performance (rel_performance=low): {len(cumulative_bottom)} circuits")
        if len(cumulative_bottom) > 0:
            for c in cumulative_bottom[:5]:  # Print only the first 5
                print(f"    Circuit {c['id']}: energy={c['energy']:.6f}")
            if len(cumulative_bottom) > 5:
                print(f"    ... and {len(cumulative_bottom) - 5} more")

        # Combine all accumulated samples
        all_ft_circuits = cumulative_top + cumulative_medium + cumulative_bottom

        # Fine-tune the design principles
        sys_prompt_ft, user_prompt_ft = build_fine_tuning_prompt(
            current_principles, all_ft_circuits,
            cumulative_top, cumulative_medium, cumulative_bottom,
            len(circuits), round_num, previous_accuracy,
            include_knowledge_base=(knowledge_base == 1),
            knowledge_base_path=knowledge_base_path
        )

        fine_tuned_principles = call_llm(sys_prompt_ft, user_prompt_ft)

        # Save the fine-tuned principles to the timestamped folder
        principles_file = os.path.join(output_folder_path, f"quantum_design_principles_v11_{round_num}.txt")
        save_fine_tuned_principles(fine_tuned_principles, principles_file, round_num)

        # Evaluate circuits with the fine-tuned principles without energy ranks
        print(f"\n[Round {round_num}] Evaluating top {EVAL_K} circuits with fine-tuned principles...")
        sys_prompt_eval, user_prompt_eval = build_evaluation_prompt(
            fine_tuned_principles, circuits, eval_k=EVAL_K
        )

        response_eval = call_llm(sys_prompt_eval, user_prompt_eval)
        selected_ids_eval = parse_selected_circuits(response_eval)[:EVAL_K]

        round_overlap, round_set = calculate_overlap_rate(set(selected_ids_eval), actual_topK)

        print(f"\n[Round {round_num} Results]")
        print(f"  Predicted top{EVAL_K}: {selected_ids_eval}")
        print(f"  Actual top{EVAL_K}: {actual_topK_sorted}")
        print(f"  ★★★ Overlap Rate: {round_overlap:.2%} ({len(round_set)}/{EVAL_K}) ★★★")

        # Unless this is the final round, have the LLM select LLM_SELECT_K new circuits without replacement
        if not is_final_round:
            print(f"\n[Round {round_num}] LLM selecting best {LLM_SELECT_K} NEW circuits for next round...")

            # Filter out circuits that have already been selected
            available_circuits = [c for c in circuits if c['id'] not in all_selected_ids]
            print(f"Available circuits for selection: {len(available_circuits)} (excluding {len(all_selected_ids)} already selected)")

            # Have the LLM select from the remaining circuits
            sys_prompt_select, user_prompt_select = build_evaluation_prompt(
                fine_tuned_principles, available_circuits, eval_k=LLM_SELECT_K
            )
            response_select = call_llm(sys_prompt_select, user_prompt_select)
            new_selected_ids = parse_selected_circuits(response_select)[:LLM_SELECT_K]

            print(f"LLM selected {len(new_selected_ids)} NEW circuits: {new_selected_ids}")

            # Add them to the cumulative selection list
            all_selected_ids.extend(new_selected_ids)
            print(f"Total accumulated circuits for next round: {len(all_selected_ids)}")

        # Calculate energy statistics
        predicted_energies = [circuits[i]['energy'] for i in selected_ids_eval if i < len(circuits)]
        avg_energy = np.mean(predicted_energies) if predicted_energies else 0
        best_energy = min(predicted_energies) if predicted_energies else 0

        print(f"\n[Round {round_num} Energy Statistics]")
        print(f"  Predicted circuits energies (sorted by energy):")
        # Sort output by energy
        sorted_predictions = sorted(
            [(cid, circuits[cid]['energy']) for cid in selected_ids_eval if cid < len(circuits)],
            key=lambda x: x[1]
        )
        for cid, energy in sorted_predictions:
            print(f"    Circuit {cid}: {energy:.6f}")
        print(f"  Average energy: {avg_energy:.6f}")
        print(f"  Best (lowest) energy: {best_energy:.6f}")

        all_results.append({
            'round': round_num,
            'principles_file': principles_file,
            'overlap_rate': round_overlap,
            'overlap_count': len(round_set),
            'predicted_ids': selected_ids_eval,
            'num_ft_samples': len(all_ft_circuits),
            'predicted_energies': predicted_energies,
            'avg_energy': avg_energy,
            'best_energy': best_energy
        })

        # Update the current principles and previous-round accuracy for the next round
        current_principles = fine_tuned_principles
        previous_accuracy = round_overlap

    # Summarize the final results
    print("\n" + "=" * 80)
    print("Final Results Summary")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  - TOP_SAMPLE_K: {TOP_SAMPLE_K}")
    print(f"  - BOTTOM_SAMPLE_K: {BOTTOM_SAMPLE_K}")
    print(f"  - RANDOM_SAMPLE_K: {RANDOM_SAMPLE_K}")
    print(f"  - TOP_NEW_K: {TOP_NEW_K}")
    print(f"  - BOTTOM_NEW_K: {BOTTOM_NEW_K}")
    print(f"  - RANDOM_NEW_K: {RANDOM_NEW_K}")
    print(f"  - Sampling strategy: LLM-guided cumulative selection (without replacement)")
    print(f"    * Baseline: selects {LLM_SELECT_K} circuits")
    print(f"    * Round 1: uses 15 circuits, selects {LLM_SELECT_K} NEW circuits")
    print(f"    * Round 2: uses 30 circuits, selects {LLM_SELECT_K} NEW circuits")
    print(f"    * Round 3+: continues cumulative pattern")
    print(f"  - knowledge_base: {knowledge_base}")
    print(f"  - EVAL_K: {EVAL_K}")
    print(f"  - LLM_SELECT_K: {LLM_SELECT_K}")
    print(f"  - NUM_ROUNDS: {NUM_ROUNDS}")
    print(f"  - Total circuits: {len(circuits)}")

    print(f"\nResults Comparison:")
    print(f"  ┌──────────┬──────────────────────────┬──────────────┬────────────┐")
    print(f"  │ Round    │ Principles File          │ Overlap Rate │ Samples    │")
    print(f"  ├──────────┼──────────────────────────┼──────────────┼────────────┤")

    for result in all_results:
        principles_name = os.path.basename(result['principles_file'])
        print(f"  │ {result['round']:6d}  │ {principles_name:24s} │ {result['overlap_rate']:.2%} ({result['overlap_count']}/{EVAL_K})   │ {result['num_ft_samples']:8d} │")

    print(f"  └──────────┴──────────────────────────┴──────────────┴────────────┘")

    print(f"\nActual lowest energy TOP_{EVAL_K} circuits (sorted by energy): {actual_topK_sorted}")

    # Print detailed energy statistics for each round
    print(f"\n" + "=" * 80)
    print("Energy Statistics for Each Round")
    print("=" * 80)
    for result in all_results:
        print(f"\nRound {result['round']}:")
        print(f"  Predicted circuit IDs: {result['predicted_ids']}")
        print(f"  Predicted energies: {[f'{e:.6f}' for e in result['predicted_energies']]}")
        print(f"  Average energy: {result['avg_energy']:.6f}")
        print(f"  Best (lowest) energy: {result['best_energy']:.6f}")

    # Plot energy trends and accuracy
    print(f"\n" + "=" * 80)
    print("Plotting Energy Trends and Accuracy Across Rounds")
    print("=" * 80)

    rounds = [r['round'] for r in all_results]
    avg_energies = [r['avg_energy'] for r in all_results]
    best_energies = [r['best_energy'] for r in all_results]
    overlap_rates = [r['overlap_rate'] * 100 for r in all_results]  # Convert to percentages

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left Y-axis: energy
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12, color='black')
    line1 = ax1.plot(rounds, best_energies, marker='o', label='Best Energy',
                     linewidth=2, markersize=8, color=color1)
    line2 = ax1.plot(rounds, avg_energies, marker='s', label='Average Energy',
                     linewidth=2, markersize=8, color=color2)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # Right Y-axis: accuracy; use a 0-110 range so the legend does not obscure 100
    ax2 = ax1.twinx()
    color3 = 'tab:green'
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color=color3)
    line3 = ax2.plot(rounds, overlap_rates, marker='^', label='Accuracy',
                     linewidth=2, markersize=8, color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.set_ylim(0, 110)  # Set the accuracy axis range to 0-110 so 100 appears below the legend

    # Combine the legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=11, loc='upper right')

    plt.title(f'Circuit Energy and Accuracy Trends by Round ({DATASET})', fontsize=14, fontweight='bold')
    ax1.set_xticks(rounds)
    fig.tight_layout()

    # Calculate improvement and find the best round; choose the earlier round when accuracy ties
    best_result = max(all_results, key=lambda x: (x['overlap_rate'], -x['round']))
    improvement = best_result['overlap_rate'] - baseline_overlap
    improvement_pct = (improvement / baseline_overlap * 100) if baseline_overlap > 0 else 0

    # Rename the folder after the experiment to include its parameters
    accuracy_pct = int(best_result['overlap_rate'] * 100)
    final_folder_name = f"run_{timestamp}_{best_result['round']}round_{accuracy_pct}_{a}_{knowledge_base}"
    final_folder_path = os.path.join(OUTPUT_DIR, final_folder_name)

    # Rename the folder
    if os.path.exists(output_folder_path) and output_folder_path != final_folder_path:
        os.rename(output_folder_path, final_folder_path)
        output_folder_path = final_folder_path
        print(f"\nFolder renamed to: {final_folder_path}")

    # Save charts to the final folder
    plot_path = os.path.join(output_folder_path, f"energy_trends_v11_{DATASET}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Energy trend plot saved to: {plot_path}")
    plt.close()

    print(f"\n" + "=" * 80)
    print("Plotting Energy Distribution Comparison")
    print("=" * 80)

    # Prepare data: predicted circuit energies versus actual top-circuit energies for each round
    fig, axes = plt.subplots(1, NUM_ROUNDS + 1, figsize=(4 * (NUM_ROUNDS + 1), 6), sharey=True)
    if NUM_ROUNDS == 0:
        axes = [axes]

    # Energies of the actual top circuits
    actual_top_energies = [circuits[i]['energy'] for i in actual_topK_sorted if i < len(circuits)]

    # Baseline (Round 0)
    baseline_predicted_ids = all_results[0]['predicted_ids']
    baseline_energies = [circuits[i]['energy'] for i in baseline_predicted_ids if i < len(circuits)]

    # Plot the baseline
    ax = axes[0]
    positions = [1, 2]
    data_to_plot = [actual_top_energies, baseline_energies]
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Actual Top', 'Predicted'], fontsize=10)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_title(f'Baseline (Round 0)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot each round
    for idx, result in enumerate(all_results[1:], start=1):  # Start at round 1 and skip the baseline
        ax = axes[idx]
        predicted_energies = result['predicted_energies']

        positions = [1, 2]
        data_to_plot = [actual_top_energies, predicted_energies]
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_xticks(positions)
        ax.set_xticklabels(['Actual Top', 'Predicted'], fontsize=10)
        ax.set_title(f'Round {result["round"]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Energy Distribution: Predicted Circuits vs. Actual Top Circuits ({DATASET})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    energy_dist_path = os.path.join(output_folder_path, f"energy_distribution_v11_{DATASET}.png")
    plt.savefig(energy_dist_path, dpi=300, bbox_inches='tight')
    print(f"Energy distribution plot saved to: {energy_dist_path}")
    plt.close()

    print(f"\n" + "=" * 80)
    print("Plotting Predicted Circuit Ranking Heatmap")
    print("=" * 80)

    # Prepare each round's predicted circuit IDs and their actual ranks
    # First, map circuit IDs to ranks
    circuit_id_to_rank = {}
    sorted_circuits = sorted(enumerate(circuits), key=lambda x: x[1]['energy'])
    for rank, (cid, _) in enumerate(sorted_circuits):
        circuit_id_to_rank[cid] = rank + 1  # Ranks start at 1

    # Build the heatmap data matrix
    heatmap_data = []
    round_labels = []
    circuit_labels = []

    # Baseline
    baseline_predicted_ids = all_results[0]['predicted_ids']
    round_labels.append('Baseline')
    baseline_ranks = [circuit_id_to_rank.get(cid, len(circuits)) for cid in baseline_predicted_ids]
    heatmap_data.append(baseline_ranks)
    circuit_labels.append([f'C{cid}' for cid in baseline_predicted_ids])

    # Each round
    for result in all_results[1:]:  # Start at round 1 and skip the baseline
        round_labels.append(f'Round {result["round"]}')
        predicted_ranks = [circuit_id_to_rank.get(cid, len(circuits)) for cid in result['predicted_ids']]
        heatmap_data.append(predicted_ranks)
        circuit_labels.append([f'C{cid}' for cid in result['predicted_ids']])

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, (NUM_ROUNDS + 1) * 0.8)))
    heatmap_array = np.array(heatmap_data)

    # Use a reversed colormap so better (lower) ranks appear darker
    im = ax.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(circuits))

    # Set the axes
    ax.set_xticks(np.arange(EVAL_K))
    ax.set_yticks(np.arange(len(round_labels)))
    ax.set_xticklabels([f'Prediction {i+1}' for i in range(EVAL_K)], fontsize=10)
    ax.set_yticklabels(round_labels, fontsize=10)

    # Show the circuit ID and rank in each cell
    for i in range(len(round_labels)):
        for j in range(EVAL_K):
            text = ax.text(j, i, f'{circuit_labels[i][j]}\n(#{heatmap_array[i, j]})',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    ax.set_title(f'Actual Energy Ranks of Predicted Circuits ({DATASET})\nGreener colors indicate better ranks', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prediction Position', fontsize=11)
    ax.set_ylabel('Round', fontsize=11)

    # Add a color bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Actual Energy Rank (1 = Best)', fontsize=10)

    plt.tight_layout()

    ranking_heatmap_path = os.path.join(output_folder_path, f"ranking_heatmap_v11_{DATASET}.png")
    plt.savefig(ranking_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Ranking heatmap saved to: {ranking_heatmap_path}")
    plt.close()

    print(f"\n" + "=" * 80)
    print("Saving Energy Values to Text File")
    print("=" * 80)

    energy_txt_path = os.path.join(output_folder_path, f"energy_values_per_round_{DATASET}.txt")
    with open(energy_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Predicted Circuit Energy Values Per Round\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"EVAL_K: {EVAL_K}\n")
        f.write(f"a = {a}\n")
        f.write(f"knowledge_base = {knowledge_base}\n")
        f.write("=" * 80 + "\n\n")

        # Write the actual top-circuit energies for reference
        f.write("Reference: Actual Top-Circuit Energies\n")
        actual_top_energies_sorted = sorted([circuits[i]['energy'] for i in actual_topK_sorted if i < len(circuits)])
        for energy in actual_top_energies_sorted:
            f.write(f"{energy:.6f}\n")
        f.write("\n")

        # Write the predicted circuit energies for each round
        for result in all_results:
            round_num = result['round']
            predicted_ids = result['predicted_ids']
            overlap_rate = result['overlap_rate']

            # Sort by energy
            sorted_predictions = sorted(
                [(cid, circuits[cid]['energy']) for cid in predicted_ids if cid < len(circuits)],
                key=lambda x: x[1]
            )

            if round_num == 0:
                f.write(f"Baseline (Round 0) - Accuracy: {overlap_rate:.0%}\n")
            else:
                f.write(f"Round {round_num} - Accuracy: {overlap_rate:.0%}\n")

            for cid, energy in sorted_predictions:
                f.write(f"{energy:.6f}\n")

            f.write("\n")  # Separate rounds with a blank line

        f.write("=" * 80 + "\n")
        f.write("End of Energy Values Report\n")
        f.write("=" * 80 + "\n")

    print(f"Energy values text file saved to: {energy_txt_path}")

    print(f"\nAll output files saved to: {output_folder_path}")

    print(f"\nBaseline Overlap Rate: {baseline_overlap:.2%}")
    print(f"Best Overlap Rate: {best_result['overlap_rate']:.2%} (Round {best_result['round']})")
    print(f"Absolute Improvement: +{improvement:.2%}")
    print(f"Relative Improvement: +{improvement_pct:.1f}%")

    if improvement > 0:
        print(f"\n✓ Fine-tuning is effective! After {NUM_ROUNDS} rounds, accuracy improved by {improvement:.2%}")
    else:
        print(f"\n✗ Fine-tuning shows no significant improvement")

    print("\n" + "=" * 80)
    print("\n【Ablation Study Explanation】")
    print("Key Differences from v9:")
    print("1. Uses relative performance labels (high/medium/low) instead of precise ranking")
    print("2. Reduces memorization risk by avoiding exact rankings")
    print("3. Three-tier system provides necessary context for learning")
    print("4. LLM-guided selection with cumulative learning:")
    print(f"   - Each round: LLM evaluates top {EVAL_K} circuits for accuracy")
    print(f"   - Non-final rounds: LLM also selects {LLM_SELECT_K} NEW circuits for next round")
    print(f"   - Cumulative: Round 1 uses 15 circuits, Round 2 uses 30, Round 3 uses 45, etc.")
    print("5. Without replacement: Each circuit selected only once across all rounds")
    print(f"6. Output files: quantum_design_principles_v11_1.txt, v11_2.txt, v11_3.txt...")

    return all_results, output_folder_path, timestamp, best_result

def main():
    """Run multiple experiments and generate averaged results."""
    print(f"\n{'='*80}")
    print(f"Quantum Circuit Design Principle Learning v1111 - Multiple Experiments")
    print(f"Running {xh_NUM} experiments")
    print(f"Dataset: {DATASET}")
    print(f"{'='*80}")

    # Store all experiment results
    all_experiments_results = []
    all_experiments_folders = []
    first_exp_info = None

    # Run all experiments
    for exp_id in range(1, xh_NUM + 1):
        print(f"\n{'='*80}")
        print(f"Starting experiment {exp_id}/{xh_NUM}")
        print(f"{'='*80}")

        exp_results, exp_folder, exp_timestamp, exp_best = run_single_experiment(exp_id)
        all_experiments_results.append(exp_results)
        all_experiments_folders.append(exp_folder)

        # Save the first experiment's information for naming the final folder
        if exp_id == 1:
            first_exp_info = {
                'timestamp': exp_timestamp,
                'best_round': exp_best['round'],
                'accuracy_pct': int(exp_best['overlap_rate'] * 100)
            }

        # Rename the plot from this experiment
        old_plot_path = os.path.join(exp_folder, f"energy_trends_v11_{DATASET}.png")
        new_plot_path = os.path.join(exp_folder, f"energy_trends_v11_{DATASET}_{exp_id}.png")
        if os.path.exists(old_plot_path):
            os.rename(old_plot_path, new_plot_path)
            print(f"Renamed plot for experiment {exp_id}: {new_plot_path}")

    # Create the aggregate folder using the first experiment's parameters
    final_folder_name = f"run_{first_exp_info['timestamp']}_{first_exp_info['best_round']}round_{first_exp_info['accuracy_pct']}_{a}_{knowledge_base}_PJ"
    final_folder_path = os.path.join(OUTPUT_DIR, final_folder_name)
    os.makedirs(final_folder_path, exist_ok=True)
    print(f"\nCreated aggregate folder: {final_folder_path}")

    # Move all experiment folders into the aggregate folder
    for exp_folder in all_experiments_folders:
        folder_name = os.path.basename(exp_folder)
        new_location = os.path.join(final_folder_path, folder_name)
        if os.path.exists(exp_folder):
            shutil.move(exp_folder, new_location)
            print(f"Moved folder: {folder_name}")

    # Generate averaged charts
    print(f"\n{'='*80}")
    print("Generating averaged experiment charts")
    print(f"{'='*80}")

    num_rounds = len(all_experiments_results[0])
    avg_best_energies = []
    avg_avg_energies = []
    avg_accuracies = []

    for round_idx in range(num_rounds):
        best_energies_this_round = [exp[round_idx]['best_energy'] for exp in all_experiments_results]
        avg_energies_this_round = [exp[round_idx]['avg_energy'] for exp in all_experiments_results]
        accuracies_this_round = [exp[round_idx]['overlap_rate'] * 100 for exp in all_experiments_results]

        avg_best_energies.append(np.mean(best_energies_this_round))
        avg_avg_energies.append(np.mean(avg_energies_this_round))
        avg_accuracies.append(np.mean(accuracies_this_round))

    rounds = list(range(num_rounds))

    # Plot averaged results
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = 'tab:blue'
    color2 = 'tab:orange'
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12, color='black')
    line1 = ax1.plot(rounds, avg_best_energies, marker='o', label='Average Best Energy',
                     linewidth=2, markersize=8, color=color1)
    line2 = ax1.plot(rounds, avg_avg_energies, marker='s', label='Average Mean Energy',
                     linewidth=2, markersize=8, color=color2)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color3 = 'tab:green'
    ax2.set_ylabel('Accuracy (%)', fontsize=12, color=color3)
    line3 = ax2.plot(rounds, avg_accuracies, marker='^', label='Average Accuracy',
                     linewidth=2, markersize=8, color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.set_ylim(0, 110)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=11, loc='upper right')

    plt.title(f'Average Trends Across {xh_NUM} Experiments ({DATASET})', fontsize=14, fontweight='bold')
    ax1.set_xticks(rounds)
    fig.tight_layout()

    plot_path = os.path.join(final_folder_path, f"energy_trends_v11_{DATASET}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Average chart saved to: {plot_path}")
    plt.close()

    # Export the results to an Excel spreadsheet
    print(f"\n{'='*80}")
    print("Exporting results to Excel")
    print(f"{'='*80}")

    # Prepare the data
    rows = []

    for exp_id, exp_results in enumerate(all_experiments_results, 1):
        # Best-energy row
        row_best = {'Experiment': f'Experiment {exp_id}', 'Metric': 'Best Energy'}
        for round_result in exp_results:
            row_best[str(round_result['round'])] = round_result['best_energy']
        rows.append(row_best)

        # Average-energy row
        row_avg = {'Experiment': f'Experiment {exp_id}', 'Metric': 'Average Energy'}
        for round_result in exp_results:
            row_avg[str(round_result['round'])] = round_result['avg_energy']
        rows.append(row_avg)

        # Accuracy row
        row_acc = {'Experiment': f'Experiment {exp_id}', 'Metric': 'Accuracy'}
        for round_result in exp_results:
            row_acc[str(round_result['round'])] = round_result['overlap_rate'] * 100
        rows.append(row_acc)

    # Calculate averages
    row_avg_best = {'Experiment': 'Experiment Average', 'Metric': 'Best Energy'}
    row_avg_avg = {'Experiment': 'Experiment Average', 'Metric': 'Average Energy'}
    row_avg_acc = {'Experiment': 'Experiment Average', 'Metric': 'Accuracy'}

    for round_idx in range(num_rounds):
        best_energies = [exp[round_idx]['best_energy'] for exp in all_experiments_results]
        avg_energies = [exp[round_idx]['avg_energy'] for exp in all_experiments_results]
        accuracies = [exp[round_idx]['overlap_rate'] * 100 for exp in all_experiments_results]

        row_avg_best[str(round_idx)] = np.mean(best_energies)
        row_avg_avg[str(round_idx)] = np.mean(avg_energies)
        row_avg_acc[str(round_idx)] = np.mean(accuracies)

    rows.append(row_avg_best)
    rows.append(row_avg_avg)
    rows.append(row_avg_acc)

    # Convert to a DataFrame
    df = pd.DataFrame(rows)

    # Save the Excel file
    excel_path = os.path.join(final_folder_path, f"experiment_results_{DATASET}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Excel spreadsheet saved to: {excel_path}")

    print(f"\n{'='*80}")
    print(f"All {xh_NUM} experiments completed!")
    print(f"Output folder: {final_folder_path}")
    print(f"{'='*80}")

    return all_experiments_results

if __name__ == "__main__":
    results = main()
