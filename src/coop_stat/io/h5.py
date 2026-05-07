"""Common H5 read/write utilities used by histogram, OVL, and compare stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def pick_3d_dataset(h5, preferred=None) -> str:
    """Find a 3-D dataset in an H5 file, preferring *preferred* if it exists."""
    if preferred and preferred in h5 and isinstance(h5[preferred], h5py.Dataset):
        return preferred
    for name in h5:
        ds = h5[name]
        if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
            return name
    raise KeyError("No 3-D dataset found in H5 file")


def read_ovl_dict(path: Path) -> Dict[str, np.ndarray]:
    """Read all non-meta datasets from an OVL H5 file.

    Skips mean/sym-map datasets that are derived, not per-pair OVL.
    """
    skip_prefixes = (
        "ovl_mean_deduped",
        "ovl_mean_eligible",
        "ovl_mean_naive",
        "ovl_mean",
        "c2_sym_map",
    )
    result = {}
    import h5py
    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            if any(key.startswith(p) or key == p for p in skip_prefixes):
                continue
            result[key] = h5[key][...]
    return result


def write_datasets(
    path: Path,
    data: Dict[str, np.ndarray],
    attrs: Optional[Dict[str, Any]] = None,
    sym_map: Optional[np.ndarray] = None,
) -> Path:
    """Write multiple datasets to an H5 file with optional attributes.

    Parameters
    ----------
    path : Path
        Output file path (parent dirs created if needed).
    data : dict
        Dataset name → numpy array.
    attrs : dict, optional
        File-level attributes.
    sym_map : ndarray, optional
        C2 symmetry map to include.

    Returns
    -------
    Path
        The written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(path, "w") as h5:
        for key, arr in data.items():
            h5.create_dataset(key, data=arr, compression="gzip", shuffle=True)
        if sym_map is not None and "c2_sym_map" not in data:
            h5.create_dataset(
                "c2_sym_map", data=sym_map, compression="gzip", shuffle=True
            )
        if attrs:
            for k, v in attrs.items():
                h5.attrs[k] = v
    return path


def read_resids(path: Path, dset_name: str = "resids") -> Optional[np.ndarray]:
    """Read the residue ID array from a distance H5 file, or None."""
    import h5py
    with h5py.File(path, "r") as h5:
        if dset_name not in h5:
            return None
        return h5[dset_name][...].astype(int)

def translate_view_label(label):
    """Translate on-disk site1/site2as1 tokens to PLmat1/PLmat2as1."""
    label = label.replace("_site2as1_", "_PLmat2as1_").replace("_site1_", "_PLmat1_")
    if label.endswith("_site2as1"):
        label = label[:-9] + "_PLmat2as1"
    elif label.endswith("_site1"):
        label = label[:-6] + "_PLmat1"
    return label


def normalize_cross_keys(raw, tag_ref, tag_cmp):
    """Normalize cross OVL dict keys: translate view labels, reorder so cmp comes first."""
    from coop_stat.compare.extraction import parse_pair_key
    out = {}
    for k, M in raw.items():
        p = parse_pair_key(k)
        if p is None:
            if "_vs_" in k:
                out[translate_view_label(k)] = M
            continue
        Lt, Lo, Rt, Ro = p
        if Lt == tag_cmp and Rt == tag_ref:
            out[f"{tag_cmp}_{translate_view_label(Lo)}_vs_{tag_ref}_{translate_view_label(Ro)}"] = M
        elif Lt == tag_ref and Rt == tag_cmp:
            out[f"{tag_cmp}_{translate_view_label(Ro)}_vs_{tag_ref}_{translate_view_label(Lo)}"] = M
    return out


def normalize_intra_keys(raw, tag):
    """Normalize intra OVL dict keys: translate view labels, filter to matching tag."""
    from coop_stat.compare.extraction import parse_pair_key
    out = {}
    for k, M in raw.items():
        p = parse_pair_key(k)
        if p is None:
            if tag in k and "_vs_" in k:
                out[translate_view_label(k)] = M
            continue
        Lt, Lo, Rt, Ro = p
        if Lt != tag or Rt != tag:
            continue
        out[f"{tag}_{translate_view_label(Lo)}_vs_{tag}_{translate_view_label(Ro)}"] = M
    return out


def load_ovl_cross(tag_ref, tag_cmp, ovl_dir):
    """Load and normalize cross OVL dict, searching standard filename candidates."""
    ovl_dir = Path(ovl_dir)
    candidates = [
        ovl_dir / f"ovl_cross_{tag_ref}_vs_{tag_cmp}.h5",
        ovl_dir / f"ovl_cross_{tag_cmp}_vs_{tag_ref}.h5",
        ovl_dir / f"ovl_results_{tag_ref}_vs_{tag_cmp}_all_replicates.h5",
        ovl_dir / f"ovl_results_{tag_cmp}_vs_{tag_ref}_all_replicates.h5",
    ]
    for c in candidates:
        if c.exists():
            raw = read_ovl_dict(c)
            return normalize_cross_keys(raw, tag_ref, tag_cmp)
    raise FileNotFoundError(
        f"No cross OVL for {tag_ref} vs {tag_cmp}. Checked: {[str(c) for c in candidates]}"
    )


def load_ovl_intra(tag, tag_ref, tag_cmp, ovl_dir):
    """Load and normalize intra OVL dict, searching standard filename candidates."""
    ovl_dir = Path(ovl_dir)
    from coop_stat.tags import parse_tag
    info = parse_tag(tag)
    core = info.core if info.kind == "single" else str(tag)
    candidates = [
        ovl_dir / f"ovl_intra_{tag}_in_{tag_cmp}_vs_{tag_ref}.h5",
        ovl_dir / f"ovl_intra_{tag}_in_{tag_ref}_vs_{tag_cmp}.h5",
        ovl_dir / f"ovl_intra_{core}_{tag_cmp}_vs_{tag_ref}.h5",
        ovl_dir / f"ovl_results_{tag}_vs_{tag}_all_replicates.h5",
    ]
    for c in candidates:
        if c.exists():
            raw = read_ovl_dict(c)
            return normalize_intra_keys(raw, tag)
    raise FileNotFoundError(
        f"No intra OVL for {tag}. Checked: {[str(c) for c in candidates]}"
    )
