"""Read .gro topology files for sequence extraction and client identification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def parse_ligand_ranges(spec: str) -> List[Tuple[int, int]]:
    """Parse ``'188-202,212-226'`` → ``[(188, 202), (212, 226)]``."""
    out = []
    for chunk in spec.split(","):
        a, b = chunk.strip().split("-")
        out.append((int(a), int(b)))
    if len(out) != 2:
        raise ValueError(f"Need exactly 2 ligand ranges, got: {spec}")
    return out


def gro_sequence_for_range(
    gro_path: Path,
    start_resid: int,
    stop_resid: int,
    aa3_to_1: Dict[str, str],
) -> str:
    """Extract one-letter sequence from a .gro file for a residue range.

    Only CA atoms are considered.  The range is inclusive on both ends.

    Parameters
    ----------
    gro_path : Path
        Path to the .gro file.
    start_resid, stop_resid : int
        Inclusive residue ID range.
    aa3_to_1 : dict
        Three-letter → one-letter amino acid mapping.

    Returns
    -------
    str
        One-letter sequence string.
    """
    seq_map: Dict[int, str] = {}
    with open(gro_path) as fh:
        lines = fh.readlines()
    if len(lines) < 3:
        raise ValueError(f"{gro_path} does not look like a .gro file")
    for line in lines[2:-1]:
        if len(line) < 20:
            continue
        try:
            resid = int(line[0:5].strip())
        except ValueError:
            continue
        resname = line[5:10].strip().upper()
        atomname = line[10:15].strip().upper()
        if atomname != "CA":
            continue
        if start_resid <= resid <= stop_resid:
            if resname not in aa3_to_1:
                raise ValueError(f"Unknown residue name '{resname}' in {gro_path}")
            seq_map[resid] = aa3_to_1[resname]
    missing = [r for r in range(start_resid, stop_resid + 1) if r not in seq_map]
    if missing:
        raise ValueError(
            f"Missing CA residues in {gro_path} for range "
            f"{start_resid}-{stop_resid}: {missing}"
        )
    return "".join(seq_map[r] for r in range(start_resid, stop_resid + 1))
