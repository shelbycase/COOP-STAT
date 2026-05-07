# COOP-STAT
🛠️ COOP-STAT is under construction 🛠️ 
As of May 7, 2026, files are operational, but debugging and testing is still underway. 

COOP-STAT (COOPerativity STatistical Analysis Toolkit) is an open-source pipeline for detecting and quantifying cooperative binding behavior in homodimeric protein–peptide systems from MD ensembles. Given singly- and doubly-bound trajectories, COOP-STAT compares internal-coordinate distributions (namely inter-residue distances) site-by-site using similarity metrics (e.g., the Overlapping Coefficient) and the two-sample Kolmogorov–Smirnov test, disentangling sampling noise (replicate-vs-replicate) from genuine ensemble differences (species-vs-species). The framework is alignment-free, designed for nuanced conformational changes, and was developed on LC8–client peptide ensembles but aims to be generalize toward other homodimeric cooperative systems.

OVL-based statistical comparison of symmetric homodimer MD simulations.

## Pipeline overview

```
distance H5s ─────────────→ histograms ──→ OVL matrices ──→ statistical comparison
(Bring your own matrix)     (dimer-hist)    (dimer-ovl)      (dimer-compare)
```

Each stage is both a CLI tool and a Python library you can import directly.

## Installation

```bash
pip install -e ".[dev]"   # editable install with test dependencies
```

## Quick start

### 1. Configure your system

Every pipeline stage needs a `DimerSystem` describing your homodimer topology.
LC8 comes preconfigured; other systems just need a chain length:

```python
from dimer_ovl import DimerSystem
from dimer_ovl.config import lc8_system

# LC8 preset
system = lc8_system()

# Custom homodimer (50 residues per chain)
system = DimerSystem(
    chain_length=50,
    peptide_seqs={"LigA": "ACDEFGHIKL", "LigB": "MNPQRSTVWY"},
    binding_windows_1=[(0, 20), (70, 90)], #note that cross-monomer pairs are allowed
    binding_windows_2=[(21, 49), (50, 69)],
)
```

### 2. Parse system tags

**Tags encode system identity and binding state:**
Naming convention follows: {Peptide Name}{Bound State Abbrev.}{Peptide Length} except for mixed client peptide systems, which follows: {Peptide 1 Name}m{Peptide Length}_{Peptide 2 Name}m{Peptide Length}
Distance Matrices that are brought in should be in the 3D format: (protein+peptide1+peptide2)x(protein+peptide1+peptide2)x(no. trajectory frames)

```python
from dimer_ovl.tags import parse_tag, dd_sym_applies

ref = parse_tag("BSN2s15")      # singly bound
cmp = parse_tag("BSN2d15")      # doubly bound (homo)
het = parse_tag("BSN2m15_SPAG5m15")  # mixed (het)

ref.sym_class       # SymClass.SINGLE
cmp.comparison_views # ["PLmat1", "PLmat2as1"]
dd_sym_applies(ref, cmp)  # True (at least one is d_homo)
```

### 3. Build histograms (from distance H5s)

```python
from dimer_ovl.histogram.builder import (
    compute_pairwise_minmax, build_edges, build_histogram
)

# Pass 1: compute shared bin edges
This must be passed for all available systems to avoid non-uniform bin edges, which would produce improper OVLs and KS-test results, and so on. 
dmin, dmax = compute_pairwise_minmax(tagged_paths, n_canon)
edges = build_edges(dmin, dmax, n_bins=60)

# Pass 2: build histograms
counts, out_edges, canon_idx, resids = build_histogram(
    tag, view, dist_h5, edges
)
```

### 4. Compute OVL matrices

```python
from dimer_ovl.ovl.core import ovl_from_hist_h5

ovl_matrix = ovl_from_hist_h5(hist_A, hist_B)
# ovl_matrix[i, j] = distributional overlap for pair (i, j)
```

### 5. Statistical comparison

```python
from dimer_ovl.compare.extraction import extract_values_for_pair
from dimer_ovl.compare.statistics import ks_pvalue, compute_stat_sign_weights

# Extract cross and intra OVL samples for pair (i, j)
cross_vals = extract_values_for_pair(cross_dict, i, j, system, dd_sym=True)
intra_vals = extract_values_for_pair(intra_dict, i, j, system, dd_sym=True)

# KS test
p = ks_pvalue(cross_vals, intra_vals)

# Full stat-sign analysis
weights = compute_stat_sign_weights(
    nres, cross_dict, iref_dict, icmp_dict,
    system=system, dd_sym_cross=True,
)
# Coming Soon... analysis code for these outputs
```

## Architecture

```
src/dimer_ovl/
├── config.py            DimerSystem — replaces all hardcoded constants
├── tags.py              Tag parsing (BSN2s15, BSN2d15, BSN2m15_SPAG5m15)
├── topology.py          C2 symmetry, binding windows, client matching
├── io/
│   ├── gro.py           .gro topology reading
│   ├── h5.py            Common H5 utilities
│   └── pdb.py           PDB B-factor writing
├── histogram/
│   ├── indexing.py       Resid-aware canonical index construction
│   ├── builder.py        Edge computation + histogram generation
│   └── sanity.py         Pre-/post-analysis sanity checks
├── ovl/
│   ├── core.py           OVL from histogram counts
│   └── aggregation.py    Block-aware per-bucket Dimer-Dimer (DD)-deduplicated means
├── compare/
│   ├── extraction.py     OVL sample collection (5 KS rules)
│   ├── statistics.py     KS tests, stat-sign weights
│   └── output.py         TSV, PDB, TXT writers
└── cli/
    ├── make_hist.py      dimer-hist CLI
    ├── calc_ovl.py       dimer-ovl CLI
    └── compare.py        dimer-compare CLI
```

### Key design decisions

**`DimerSystem` replaces globals.** The original scripts used module-level
constants (`DIMER_N=178`, `BINDING_WINDOW_1`, `PEPTIDE_SEQS`, etc.).
Now these are fields on a dataclass that gets passed through
the pipeline. This makes the package work for any symmetric homodimer.

**Tags are structured objects.** Instead of passing raw strings and
re-parsing everywhere, `parse_tag()` returns a `Tag` dataclass with
`.sym_class`, `.comparison_views`, `.peptide_length` etc.

**Pure functions over methods.** Core computations (`ovl_from_counts`,
`ks_pvalue`, `extract_values_for_pair`) are free functions that take
data + config as arguments. Easy to test, easy to compose.

**I/O separated from logic.** H5/GRO/PDB reading is in `dimer_ovl.io`;
computation modules never touch the filesystem directly (they receive
arrays).

## Testing

```bash
pytest                    # run all tests
pytest -x                 # stop on first failure
pytest tests/test_tags.py # run specific module
pytest -k "TestC2"        # run tests matching pattern
pytest --cov=dimer_ovl    # with coverage report
```

### Test categories

- **Unit tests** (`test_tags.py`, `test_topology.py`): Pure logic, no I/O.
- **Component tests** (`test_histogram/`, `test_ovl/`, `test_compare/`):
  Use synthetic H5 fixtures from `conftest.py`.
- **Integration tests**: Wire together multiple stages with synthetic data.

## Adapting for a new system

1. Create a `DimerSystem` with your chain length, peptide sequences, and
   binding windows.
2. Tag your simulations following the naming convention:
   `<client><state><peplen>` (e.g. `MYPROTs12`, `MYPROTd12`).
3. Run the pipeline stages with your `DimerSystem` instance.

The only assumptions are:
- **Symmetric homodimer**: two identical chains related by C2 rotation.
- **Equal chain lengths**: chain A and chain B have the same number of residues.
- **Peptide ligands have highest resids**: in distance H5 files, the 2×peplen
  atoms with the highest residue IDs are the two bound peptides. This organizes the distance matrix as (protein+peptide1+peptide2)x(protein+peptide1+peptide2).
