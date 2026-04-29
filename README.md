# PSH (Phase Hunter)

**PSH（Phase Hunter）** is a Python-based toolkit for crystal structure scanning and phase-transition candidate exploration.  
It is designed to generate symmetry-broken distorted structures from a parent crystal structure and to support systematic searches for possible low-symmetry phases.

The main entry script is:

```bash
run_phase_hunter.py
```

---

## Overview

PSH is intended for research workflows involving:

- crystal phase exploration
- symmetry-breaking distortion scans
- Landau-mode-inspired structural perturbations
- high-symmetry k-point mode generation
- distorted structure reduction
- space-group-based output organization
- SLURM-based high-throughput execution

It is especially useful for first-principles studies where one wants to explore possible hidden, metastable, or symmetry-lowered phases derived from a high-symmetry parent structure.

---

## Project Structure

```text
PSH/
├── README.md
├── run_phase_hunter.py
└── phase_hunter/
    ├── config.py          # Default parameters, CLI overrides, and derived configs
    ├── cli.py             # Command-line interface and SLURM-related branches
    ├── scan_engine.py     # Core single/combo scan execution
    ├── persistence.py     # Output writing: JSONL, CSV, POSCAR, checkpoint
    └── runtime_io.py      # Runtime directory and log management
    └── ...
```

---

## Requirements

Python 3.10 or later is recommended.

Core dependencies:

```bash
python -m pip install numpy scipy spglib seekpath
```

Required packages:

- `numpy`
- `scipy`
- `spglib`
- `seekpath`

---

## Quick Start

### 1. Show help message

```bash
python run_phase_hunter.py --help
```

### 2. Run locally

```bash
python run_phase_hunter.py
```

By default, the program reads the parent structure defined in the configuration file and starts the scan using the default settings.

### 3. Specify a parent structure

```bash
python run_phase_hunter.py --parent-poscar parent.vasp
```

### 4. Specify an output directory

```bash
python run_phase_hunter.py --output-dir phase_scan_results
```

---

## SLURM Usage

PSH supports generating and submitting SLURM jobs.

### Generate SLURM script only

```bash
python run_phase_hunter.py --write-slurm
```

### Generate and submit SLURM job

```bash
python run_phase_hunter.py --submit-slurm
```

---

## Common Command-Line Options

### Input and Output

| Option | Description |
|---|---|
| `--parent-poscar PATH` | Path to the parent POSCAR/VASP structure |
| `--output-dir PATH` | Directory for scan outputs |

Example:

```bash
python run_phase_hunter.py \
    --parent-poscar parent.vasp \
    --output-dir results
```

---

### Structure Dimensionality and k-Point Selection

| Option | Description |
|---|---|
| `--structure-dimensionality {2d,3d}` | Specify whether the structure is treated as 2D or 3D |
| `--high-symmetry-kpoint-selection {path_endpoints,all_point_coords,labels}` | Method for selecting high-symmetry k points |
| `--high-symmetry-kpoint-labels "K1,K2,..."` | Manually select high-symmetry labels |
| `--exclude-gamma-high-symmetry` | Exclude the Γ point from selected high-symmetry k points |

Example:

```bash
python run_phase_hunter.py \
    --structure-dimensionality 3d \
    --high-symmetry-kpoint-selection path_endpoints
```

---

### Failure Policy

| Option | Description |
|---|---|
| `--failure-policy strict` | Stop immediately when an error occurs |
| `--failure-policy debug` | Provide more debugging information |
| `--failure-policy permissive` | Continue when non-critical failures occur |

Example:

```bash
python run_phase_hunter.py --failure-policy strict
```

---

### Distorted Structure Reduction

| Option | Description |
|---|---|
| `--reduce-distorted-symprec FLOAT` | Symmetry tolerance used for reducing distorted structures |
| `--reduce-distorted-skip-if-amplitude-below FLOAT` | Skip reduction when the distortion amplitude is below this threshold |

Example:

```bash
python run_phase_hunter.py \
    --reduce-distorted-symprec 0.1 \
    --reduce-distorted-skip-if-amplitude-below 1e-4
```

---

## Debugging

PSH provides several debugging options for checking the workflow without launching a full scan.

### Enable debug mode

```bash
python run_phase_hunter.py --debug
```

### Stop after a specific stage

```bash
python run_phase_hunter.py \
    --debug \
    --debug-stop-after-stage config
```

Available stages include:

```text
config
paths
parent
symmetry
basis
plan
single
combo
summary
```

### Disable parallel execution

```bash
python run_phase_hunter.py \
    --debug \
    --debug-no-parallel
```

### Limit the number of trials

```bash
python run_phase_hunter.py \
    --debug \
    --debug-max-trials 20
```

### Print configuration and scan plan

```bash
python run_phase_hunter.py \
    --debug \
    --debug-print-config \
    --debug-print-plan
```

---

## Output Files

A typical output directory may contain:

```text
output/
├── scan_results.jsonl
├── scan_results.csv
├── checkpoint.json
├── run.log
├── structures_by_sg/
├── hit_structures_by_sg/
└── *.structure_metadata.json
```

### Main Output Files

| File or Directory | Description |
|---|---|
| `scan_results.jsonl` | Main scan results in JSON Lines format |
| `scan_results.csv` | Optional CSV summary of scan results |
| `checkpoint.json` | Checkpoint file for restart or continuation |
| `run.log` | Runtime log file |
| `structures_by_sg/` | Generated structures grouped by space group |
| `hit_structures_by_sg/` | Candidate structures grouped by space group |
| `*.structure_metadata.json` | Metadata for generated or reduced structures |

---

## Typical Workflow

A typical PSH workflow is:

```text
1. Prepare parent POSCAR structure
2. Configure scan parameters in phase_hunter/config.py
3. Run run_phase_hunter.py
4. Generate distorted structures
5. Reduce distorted cells when applicable
6. Identify space groups
7. Write structures and metadata
8. Analyze candidate phases
```

Example command:

```bash
python run_phase_hunter.py \
    --parent-poscar parent.vasp \
    --output-dir scan_output \
    --failure-policy strict \
    --debug-print-plan
```

---

## Common Problems

### Missing dependencies

If the program reports missing packages, check the current Python environment:

```bash
python -m pip show numpy scipy spglib seekpath
```

Then install missing packages:

```bash
python -m pip install numpy scipy spglib seekpath
```

---

### Only check configuration without running a full scan

```bash
python run_phase_hunter.py \
    --debug \
    --debug-stop-after-stage config
```

---

### Reduce I/O for quick testing

For quick debugging, consider:

- using a smaller scan profile
- reducing the number of trials
- disabling CSV output
- increasing flush intervals
- increasing print intervals

For example, modify the following options in `phase_hunter/config.py`:

```python
WRITE_RESULTS_CSV = False
FLUSH_EVERY_N_RECORDS = 100
PRINT_EVERY_N_TRIALS = 100
```

---

## Notes

- The recommended entry point is `run_phase_hunter.py`.
- Most default parameters are configured in `phase_hunter/config.py`.
- For high-throughput calculations, SLURM mode is recommended.
- For debugging, use `--debug` together with `--debug-stop-after-stage`.

---

## Suggested Repository Description

```text
A symmetry-driven crystal phase exploration toolkit for generating and screening distorted structures from parent phases.
```

---

## License

Please add a license before public release, for example:

- MIT License
- Apache License 2.0
- GPLv3

---

## Citation

If this code is used in academic work, please cite the corresponding paper or repository once available.

```text
Citation information will be added here.
```
