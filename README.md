[README.md](https://github.com/user-attachments/files/29752737/README.md)
# DILAN

**DILAN** (*Differential Local Alignment of Networks*) is a rewiring-aware pairwise local alignment algorithm for differential biological networks.

DILAN identifies compact conserved modules across two differential networks by integrating node rewiring similarity, differential-edge similarity, edge conservation, greedy module expansion, and redundancy filtering.

## Main idea

Traditional network alignment methods are designed to compare static interaction networks. DILAN instead operates on **differential networks**, where edges represent condition-specific changes in interaction strength or presence.

Given two differential networks, DILAN searches for local modules that preserve similar rewiring behavior rather than requiring global network similarity.

## Features

- Pairwise local alignment of differential biological networks
- Shared-node alignment using common node identifiers
- Rewiring-aware seed selection
- Seed-centered local alignment graph construction
- Greedy module expansion
- Differential rewiring metrics:
  - DEP: Differential Edge Preservation
  - RCS: Rewiring Conservation Score
  - RNC: Rewiring Node Consistency
  - RMC: Rewiring Module Coherence
- DILAN Score for module ranking
- Redundancy filtering of highly overlapping modules
- Text and tabular output files for downstream analysis

## Repository structure

```text
DILAN/
├── src/
│   └── dilan.py
├── data/
│   └── example input files can be placed here
├── output/
│   └── DILAN results are written here
├── example/
│   └── example commands or toy datasets can be placed here
├── requirements.txt
├── README.md
├── LICENSE
└── CITATION.cff
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mmilano87/DILAN.git
cd DILAN
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

DILAN requires Python 3.9 or later.

## Input files

DILAN requires two differential networks and two node-level rewiring-score files.

### Differential edge file

Each edge file must be whitespace-separated and contain at least three columns:

```text
Node1 Node2 Weight
```

Example:

```text
UniRef90_A0A123 UniRef90_B0B456 0.742
UniRef90_C0C789 UniRef90_D0D012 0.318
```

Additional columns are ignored.

### Rewiring-score file

Each score file must contain node-level rewiring scores. The preferred format is:

```text
Node Rewiring_score
```

Example:

```text
UniRef90_A0A123 0.81
UniRef90_B0B456 0.64
```

Headerless two-column files are also supported.

## Running DILAN

Example command:

```bash
python src/dilan.py \
  --network-1-name IBD \
  --network-1-edges data/IBD_edges.txt \
  --network-1-scores data/IBD_rewiring_scores.txt \
  --network-2-name T2D \
  --network-2-edges data/T2D_edges.txt \
  --network-2-scores data/T2D_rewiring_scores.txt \
  --output-dir output/IBD_vs_T2D
```

Optional parameters:

```bash
--top-seeds 100
--min-module-size 5
--max-module-size 80
--min-module-score 0.30
--max-overlap 0.85
```

## Output files

DILAN creates the following files in the selected output directory.

### `dilan_seed_list.txt`

Ranked list of candidate seeds, including rewiring similarity, average rewiring activity, and seed score.

### `rewiring_alignment_graph_edges.txt`

All local alignment graph edges generated across all selected seeds.

### `dilan_modules.txt`

Human-readable description of the final non-redundant modules, including module nodes, alignment edges, and rewiring-aware metrics.

### `module_statistics.txt`

Tabular summary of all final modules. Each row corresponds to one module.

### `summary_alignment.txt`

One-row global summary of the DILAN run.

### `final_alignment_metrics_report.txt`

Detailed text report containing module-level metrics and global averages.

## DILAN score

For each candidate module, DILAN computes:

```text
Score(M) = 0.40 * RCS(M) + 0.30 * RNC(M) + 0.30 * RMC(M)
```

where:

- `RCS` measures differential-edge similarity.
- `RNC` measures node-level rewiring consistency.
- `RMC` measures the average local alignment strength inside the module.

`DEP` is reported as an independent topological validation metric.

## Example workflow

1. Build two differential networks from condition-specific biological networks.
2. Compute node-level rewiring scores for each network.
3. Run DILAN using the two edge files and the two rewiring-score files.
4. Inspect `module_statistics.txt` and `dilan_modules.txt`.
5. Use the detected modules for downstream biological validation, such as pathway enrichment analysis.

## Citation

If you use DILAN in your research, please cite:

```text
Milano M., Defilippo A., Veltri P., Guzzi P.H.
DILAN: Differential Local Alignment of Networks.
```

A complete citation will be added after publication.

## Code availability

The source code is available at:

```text
https://github.com/mmilano87/DILAN
```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

Marianna Milano  
University Magna Graecia of Catanzaro  
GitHub: https://github.com/mmilano87
