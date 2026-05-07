"""PDB reading and writing for statistical significance output."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from coop_stat.config import DimerSystem


def pdb_to_h5_index(line: str, system: DimerSystem) -> Optional[int]:
    """Map an ATOM/HETATM PDB line to a 0-based H5 matrix index.

    Uses chain ID or segment ID to assign chain A/B,
    then converts PDB resid (1-based) to 0-based index.
    """
    if not line.startswith(("ATOM", "HETATM")) or len(line) < 26:
        return None
    try:
        res = int(line[22:26])
    except ValueError:
        return None

    chain_id = line[21].strip() if len(line) > 21 else ""
    seg = line[72:76].strip() if len(line) >= 76 else ""

    chain = None
    if chain_id in ("A", "B"):
        chain = chain_id
    elif seg.endswith("A"):
        chain = "A"
    elif seg.endswith("B"):
        chain = "B"
    if chain is None:
        return None

    clen = system.chain_length
    mx = system.dimer_n

    if chain == "A":
        idx = res - 1 if 1 <= res <= clen else None
    else:
        idx = (
            clen + (res - 1) if 1 <= res <= clen
            else (res - 1 if clen + 1 <= res <= mx else None)
        )
    return idx if idx is not None and 0 <= idx < mx else None


def write_bfactor_pdb(
    template: Path,
    weights: np.ndarray,
    output: Path,
    system: DimerSystem,
) -> Path:
    """Write a PDB with per-residue weights in the B-factor column.

    Coordinates are copied from *template*; B-factors are replaced
    with ``max(weights[idx], 0.0)`` where *idx* is the H5 index.

    Raises RuntimeError if coordinates don't match after writing.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(template) as inp, open(output, "w") as out:
        for line in inp:
            if line.startswith(("ATOM", "HETATM")):
                idx = pdb_to_h5_index(line, system)
                if idx is not None and idx < len(weights):
                    w = max(float(weights[idx]), 0.0)
                    line = line[:60] + f"{w:6.3f}" + line[66:]
            out.write(line)
    _verify_coords(template, output)
    return output


def _verify_coords(template: Path, output: Path):
    """Assert that ATOM coordinates are byte-identical between files."""
    def _xyz(p):
        coords = []
        with open(p) as fh:
            for ln in fh:
                if ln.startswith(("ATOM", "HETATM")):
                    coords.append((
                        float(ln[30:38]),
                        float(ln[38:46]),
                        float(ln[46:54]),
                    ))
        return coords

    t_xyz, o_xyz = _xyz(template), _xyz(output)
    if len(t_xyz) != len(o_xyz):
        raise RuntimeError(
            f"Atom count mismatch: template={len(t_xyz)} output={len(o_xyz)}"
        )
    for (xt, yt, zt), (xo, yo, zo) in zip(t_xyz, o_xyz):
        if not (xt == xo and yt == yo and zt == zo):
            raise RuntimeError("Coordinate mismatch between template and output PDB")
