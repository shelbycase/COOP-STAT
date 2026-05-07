"""
Output writers for comparison results.

- Emit-stats TSV: per-pair OVL values, means, KS p-values.
- Stat-sign PDB: per-residue significance weights in B-factor column.
- Stat-sign TXT: plain-text table of residue weights.

Usage:
- see analysis code for example plot using TSV
- see instructions for analyzing PDB using PyMol (desktop)
- see analysis code for example plot using TXT
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from dimer_ovl.config import DimerSystem
from dimer_ovl.io.pdb import write_bfactor_pdb


def write_stat_sign_pdb(
    template_pdb: Path,
    weights: np.ndarray,
    tag_ref: str,
    tag_cmp: str,
    output_dir: Path,
    system: DimerSystem,
) -> Path:
    """Write a PDB with stat-sign weights in B-factor column."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{tag_cmp}_vs_{tag_ref}_stat_sign.pdb"
    return write_bfactor_pdb(template_pdb, weights, out, system)


def write_stat_sign_txt(
    weights: np.ndarray,
    tag_ref: str,
    tag_cmp: str,
    output_dir: Path,
    system: DimerSystem,
    mbw: int = 0,
    matched_client: Optional[str] = None,
    window_roles: Optional[Dict] = None,
) -> Path:
    """Write a plain-text file of per-residue significance weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{tag_cmp}_vs_{tag_ref}_stat_sign_residue_weights.txt"
    bwl = str(mbw) if mbw in (1, 2) else "both"
    with open(out, "w") as fh:
        fh.write(f"# matched_bw={bwl}\n")
        fh.write(f"# matched_client={matched_client or 'none'}\n")
        fh.write(f"# BINDING_WINDOW_1={system.binding_windows_1}\n")
        fh.write(f"# BINDING_WINDOW_2={system.binding_windows_2}\n")
        if window_roles:
            for bw in (1, 2):
                r = window_roles.get(bw, {})
                fh.write(
                    f"# BW{bw}: ref={r.get('ref_occupant') or 'unbound'} "
                    f"cmp={r.get('cmp_occupant') or 'unbound'} "
                    f"inc={r.get('included', '?')} role={r.get('role', '?')}\n"
                )
        fh.write("H5_res\tWeight\tBindingWindow\n")
        for i in range(len(weights)):
            fh.write(
                f"{i}\t{max(weights[i], 0.0)}\t"
                f"{system.residue_binding_window(i)}\n"
            )
    return out


def write_emit_stats_tsv(
    rows: List[Dict],
    output_path: Path,
    cross_labels: List[str],
    iref_labels: List[str],
    icmp_labels: List[str],
) -> Path:
    """Write the emit-stats TSV.

    Each row dict must have keys: i, j, cm, cs, rm, rs, mm, ms, kr, km, kc,
    and optionally cross_vals, iref_vals, icmp_vals (dicts of label→value).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = (
        ["Rank", "Pair(i:j)", "Cross_Mean", "Cross_SD"]
        + [f"Cross[{k}]" for k in cross_labels]
        + ["IntraPDB1_Mean", "IntraPDB1_SD"]
        + [f"IntraPDB1[{k}]" for k in iref_labels]
        + ["IntraPDB2_Mean", "IntraPDB2_SD"]
        + [f"IntraPDB2[{k}]" for k in icmp_labels]
        + ["KS_p_cross_vs_intraPDB1", "KS_p_cross_vs_intraPDB2", "KS_p_conservative"]
    )

    def _f(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None else "NA"

    with open(output_path, "w") as fh:
        fh.write("\t".join(hdr) + "\n")
        for rank, row in enumerate(rows, 1):
            line = [
                str(rank),
                f"{row['i']}:{row['j']}",
                _f(row["cm"]),
                _f(row["cs"]),
            ]
            cv = row.get("cross_vals", {})
            line.extend(_f(cv.get(k)) for k in cross_labels)
            line.extend([_f(row["rm"]), _f(row["rs"])])
            rv = row.get("iref_vals", {})
            line.extend(_f(rv.get(k)) for k in iref_labels)
            line.extend([_f(row["mm"]), _f(row["ms"])])
            mv = row.get("icmp_vals", {})
            line.extend(_f(mv.get(k)) for k in icmp_labels)
            line.extend([
                _f(row["kr"], ".6g"),
                _f(row["km"], ".6g"),
                _f(row["kc"], ".6g"),
            ])
            fh.write("\t".join(line) + "\n")
    return output_path
