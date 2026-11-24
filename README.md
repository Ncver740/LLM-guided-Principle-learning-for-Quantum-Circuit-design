# Principle-Guided Quantum Circuit Generation via Large Language Models

This repository accompanies the manuscript **"Principle-Guided Quantum Circuit Generation via Large Language Models"**, submitted to *Advanced Quantum Technologies*. It studies quantum architecture search for variational quantum eigensolvers (VQE) on Heisenberg, transverse-field Ising (TFIM), and transverse-field cluster (TFCM, named `TFCluster` in the data scripts) tasks.

The workflow transfers circuit design knowledge from 4 to 8 qubits:

1. Compare low-energy, high-energy, and intermediate reference circuits to extract principles from labeled 4-qubit data.
2. Refine the principles on 8-qubit circuits using high/medium/low relative performance labels.
3. Generate new circuits from the refined principles, optionally augmented with VQE domain knowledge, and evaluate their energies.

The paper describes three refinement rounds with 15 newly evaluated circuits per round. The released `step2_finetune_8bit_principles.py` currently sets `NUM_ROUNDS = 8` and `xh_NUM = 2`; adjust these constants for the experiment being reproduced. Root scripts use `YOUR_PATH` placeholders and require local configuration before execution.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`4-qubit/`](4-qubit/) | Sample and VQE-label 4-qubit source circuits |
| [`8-qubit/`](8-qubit/) | Sample and VQE-label 8-qubit target circuits |
| [`step1_learn_4bit_principles.py`](step1_learn_4bit_principles.py) | Extract principles by three-group comparison |
| [`step2_finetune_8bit_principles.py`](step2_finetune_8bit_principles.py) | Refine principles and measure top-k overlap |
| [`build_knowledge_base.py`](build_knowledge_base.py) | Collect optional VQE reference material |
| [`step3_generate_circuits.py`](step3_generate_circuits.py) | Generate and evaluate circuits from principles |
| [`step3_generate_circuits_with_kb.py`](step3_generate_circuits_with_kb.py) | Generate and evaluate circuits with principles and a knowledge base |

The generated circuit representation uses `['rx'/'ry'/'rz', qubit]` and `['cx'/'cz', qubit1, qubit2]`. Run sampling and labeling from the corresponding qubit-count directory; run the three root-level stages from the repository root.

## Environment

```text
numpy==1.26.4
networkx==3.1
matplotlib==3.7.2
pennylane==0.42.3
qiskit==2.3.0
tensorflow==2.20.0
tensorcircuit-ng==1.5.0
openai==2.8.1
google-genai==1.47.0
anthropic==0.91.0
pandas==2.2.3
openpyxl==3.1.5
tqdm==4.62.0
requests==2.32.4
beautifulsoup4==4.12.3
```

LLM stages need network access and credentials for an OpenAI-compatible chat-completions endpoint. Set `API_KEY`, `BASE_URL`, and `MODEL` in each relevant root script.

## Prepare labeled circuits

The release does not include sampled `samples.pkl`, optimized `energy.pkl`, learned principles, or experiment results. Only load trusted pickle files. For each task, circuit and energy lists must be aligned by index.

For example, from `4-qubit/`:

```bash
python sampling.py --task Heisenberg_4 --device_name full_4q --num_layer 10 --sample_num 110000 --seed 0
python vqe_task_paralell_Heisenberg.py --seed 0
```

This writes `cir_sample/full_4q_Heisenberg_4/training/run_0/samples.pkl` and `energy.pkl`. Use `TFIM_4` or `TFCluster_4` with the matching VQE script for the other models. Original batch examples are in `4-qubit/scripts/`.

For an 8-qubit Heisenberg sample, run from `8-qubit/`:

```bash
python sampling.py --task Heisenberg_8 --device_name grid_16q --num_layer 10 --sample_num 201000 --seed 0
```

The sampler writes under `result/cir_sample/grid_16q_Heisenberg_8/`, while the no-noise VQE scripts read from `result_no_noise/cir_sample/`. Place the intended `training/run_0/samples.pkl` in the corresponding VQE input directory before running, for example, `python vqe_task_paralell_Heisenberg.py --seed 0`. Half-noise scripts use `result_half_noise/cir_sample/`. Select the matching Heisenberg, TFIM, or TFCluster trainer and check its input path. Base-script full-noise and no-noise runs can use the same result filenames; keep those runs in separate working directories. The example sampling sizes and VQE jobs are computationally substantial.

## Run the three stages

### 1. Extract 4-qubit principles

In `step1_learn_4bit_principles.py`, choose `DATASET` from `Heisenberg`, `TFIM`, or `TFCM`. Fill its `SAMPLES_PATH`, `ENERGY_PATH`, `OUTPUT_DIR`, and `OUTPUT_PATH`, plus the LLM settings. The current code compares the 50 lowest-energy, 50 highest-energy, and 50 randomly selected middle-group circuits.

```bash
python step1_learn_4bit_principles.py
```

The initial UTF-8 principles text is written to `OUTPUT_PATH`. Group sizes and the random seed are configurable script constants.

### 2. Refine on 8-qubit circuits

In `step2_finetune_8bit_principles.py`, select the same `DATASET` and set `SAMPLES_PATH`, `ENERGY_PATH`, `PRINCIPLES_PATH` (the stage-1 text file), and `OUTPUT_DIR`. Set `knowledge_base = 0` to omit the optional knowledge base, or set it to `1` and fill `knowledge_base_path`. Review `NUM_ROUNDS`, `xh_NUM`, `LLM_SELECT_K`, and `EVAL_K`.

```bash
python step2_finetune_8bit_principles.py
```

The script saves per-round `quantum_design_principles_v11_<round>.txt` files, plots, and an aggregate Excel report under timestamped output directories. It measures top-k overlap against the labeled pool. Principle-refinement prompts use relative performance labels rather than exact energies; local energies still define those labels and evaluation statistics.

### 3. Generate and evaluate circuits

Choose one generator. Fill its API settings and the selected model's `principles_path` and `output_dir`. For the knowledge-base version, also change the `knowledge_base_path` assignment inside `main()` to an existing text file.

```bash
python step3_generate_circuits.py --model heisenberg --top_k 5
python step3_generate_circuits_with_kb.py --model heisenberg --top_k 5 --prompt_strategy uniform
```

Run **one** command per experiment. Both accept `--model heisenberg|tfim|tfcm`; the knowledge-base version also accepts `--prompt_strategy uniform|individual`. Each generates circuits, optimizes them with TensorCircuit/TensorFlow VQE, and writes `generated_samples.pkl`, `generated_energy.pkl`, `full_results.pkl`, and `metadata.pkl` under `output_dir/run_<timestamp>/`.

Both generators default to noise enabled. Their `--noise` argument uses `type=bool`, so `--noise False` still evaluates as true; change the parser default in a local copy for a no-noise generation run. The plain generator falls back to generic built-in principles when its configured file is missing; point it at learned principles for the intended workflow.

## Optional knowledge base

`python build_knowledge_base.py` fetches PennyLane/VQE pages and writes `knowledge_base3/pennylane_vqe_knowledge_base.txt` and metadata. It needs internet access; optional LLM enhancement is selected through an interactive prompt. Review retrieved content before using it as prompt context. The knowledge-base generator requires its path to be configured explicitly.
