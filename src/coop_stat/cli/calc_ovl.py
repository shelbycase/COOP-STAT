"""CLI: ``dimer-ovl`` — compute OVL matrices from histogram H5 files.

Modes:
    pair  — cross-comparisons between tag1 and tag2, plus intra for each.
    intra — intra-comparisons within tag1 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from coop_stat.config import DimerSystem, lc8_system
from coop_stat.io.h5 import pick_3d_dataset, write_datasets
from coop_stat.ovl.aggregation import build_means_dict, compute_block_aware_mean
from coop_stat.ovl.core import ovl_from_hist_h5
from coop_stat.tags import Tag, parse_tag


# ── CLI ──────────────────────────────────────────────────────────────


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute OVL matrices from histogram H5 files."
    )
    p.add_argument("tag1", help="Reference tag (e.g. BSN2s15)")
    p.add_argument("tag2", nargs="?", default=None,
                   help="Comparison tag (required for --mode pair)")
    p.add_argument("--nrep", type=int, required=True)
    p.add_argument("--mode", choices=["pair", "intra"], default="pair")
    p.add_argument("--ovl-dir", default="ovl_h5")
    p.add_argument("--hist-dir", default="hist_h5")
    p.add_argument("--hist-dset", default="hist_counts")
    p.add_argument("--edges-dset", default="bin_edges")
    p.add_argument("--dimer-n", type=int, default=None)
    p.add_argument("--dd-atol", type=float, default=1e-6)
    p.add_argument("--json-log-dir", default="ovl_h5/logs")
    p.add_argument("--skip-intra-bundles", action="store_true")
    p.add_argument("--system-preset", default="lc8",
                   choices=["lc8", "custom"])
    p.add_argument("--chain-length", type=int, default=None)
    p.add_argument("--dt", default=None)
    p.add_argument("--top-dir", default=".")
    p.add_argument("--ligand-ranges", default="188-202,212-226")
    return p.parse_args()


# ── Histogram path resolution ────────────────────────────────────────

_VIEW_TO_FILE = {"PLmat1": "site1", "PLmat2as1": "site2as1", "native": "native"}


def _hist_path(hist_dir: str, tag: Tag, rep: int, view: str) -> Path:
    file_token = _VIEW_TO_FILE.get(view, view)
    base = Path(hist_dir) / str(tag)
    if file_token == "native":
        patterns = [
            base / f"{tag}_{rep}_diststart1667_hist60bins.h5",
            base / f"{tag}_{rep}_hist60bins.h5",
        ]
    else:
        patterns = [
            base / f"{tag}_{rep}_{file_token}_diststart1667_hist60bins.h5",
            base / f"{tag}_{rep}_{file_token}_hist60bins.h5",
        ]
    for p in patterns:
        if p.exists():
            return p
    from glob import glob
    g = (glob(str(base / f"{tag}_{rep}_{file_token}*.h5")) if file_token != "native"
         else glob(str(base / f"{tag}_{rep}_*.h5")))
    if g:
        return Path(sorted(g)[-1])
    raise FileNotFoundError(
        f"No histogram H5 for {tag} rep={rep} view={view}. Checked: {patterns}"
    )


# ── Object construction ──────────────────────────────────────────────


def _build_objects(tag: Tag, nrep: int, hist_dir: str) -> List[Dict]:
    objs = []
    for rep in range(1, nrep + 1):
        for view in tag.comparison_views:
            path = _hist_path(hist_dir, tag, rep, view)
            client = tag.core if tag.kind == "single" else (
                tag.core1 if view in ("PLmat1", "site1") else tag.core2
            )
            label = f"r{rep}_{view}_{client}"
            objs.append({
                "rep": rep, "view": view, "tag": str(tag), "path": path,
                "client": client, "label": label,
                "is_mixed": tag.is_mixed, "state": tag.state.value,
            })
    return objs


def _should_compare(a: Dict, b: Dict) -> bool:
    if not a["is_mixed"] and not b["is_mixed"]:
        return True
    return a["client"] == b["client"]


def _vp_label(a: Dict, b: Dict) -> str:
    return f"{a['view']}({a['client']})-vs-{b['view']}({b['client']})"


# ── Comparison loops ─────────────────────────────────────────────────


def _run_intra(tag, objs, system, hist_dset, edges_dset):
    dn = system.dimer_n
    ovl_dict, records = {}, []
    rp_first = {}
    for a, b in combinations(objs, 2):
        if a["rep"] == b["rep"]:
            if not (a["view"] != b["view"] and not a["is_mixed"]):
                continue
        if not _should_compare(a, b):
            continue
        key = f"{tag}_{a['label']}_vs_{tag}_{b['label']}"
        print(f"  [intra] {key}")
        ovl = ovl_from_hist_h5(a["path"], b["path"],
                               counts_dset=hist_dset, edges_dset=edges_dset)
        ovl_dict[key] = ovl
        if a["rep"] != b["rep"]:
            records.append({
                "key": key, "rep_A": a["rep"], "rep_B": b["rep"],
                "view_A": a["view"], "view_B": b["view"],
                "view_pair_label": _vp_label(a, b),
                "client_A": a["client"], "client_B": b["client"], "ovl": ovl,
            })
            rp = (a["rep"], b["rep"])
            if rp not in rp_first:
                rp_first[rp] = ovl
    for (rA, rB), first in rp_first.items():
        dk = f"{tag}_r{rA}_dimer_vs_{tag}_r{rB}_dimer"
        ovl_dict[dk] = first[:dn, :dn].astype(np.float32, copy=True)
    return ovl_dict, records


def _run_cross(tag1, tag2, objs1, objs2, system, hist_dset, edges_dset):
    dn = system.dimer_n
    ovl_dict, records = {}, []
    rp_first = {}
    for o2 in objs2:
        for o1 in objs1:
            if (o1["state"] == "d" and not o1["is_mixed"]
                    and o2["state"] == "d" and not o2["is_mixed"]
                    and (o1["view"] == "PLmat2as1" or o2["view"] == "PLmat2as1")):
                continue
            if not _should_compare(o1, o2):
                continue
            key = f"{tag1}_{o1['label']}_vs_{tag2}_{o2['label']}"
            print(f"  [cross] {key}")
            ovl = ovl_from_hist_h5(o1["path"], o2["path"],
                                   counts_dset=hist_dset, edges_dset=edges_dset)
            ovl_dict[key] = ovl
            records.append({
                "key": key, "rep_A": o1["rep"], "rep_B": o2["rep"],
                "view_A": o1["view"], "view_B": o2["view"],
                "view_pair_label": _vp_label(o1, o2),
                "client_A": o1["client"], "client_B": o2["client"], "ovl": ovl,
            })
            rp = (o1["rep"], o2["rep"])
            if rp not in rp_first:
                rp_first[rp] = ovl
    if not ovl_dict:
        raise RuntimeError(f"No cross comparisons for {tag1} vs {tag2}.")
    for (rA, rB), first in rp_first.items():
        dk = f"{tag1}_r{rA}_dimer_vs_{tag2}_r{rB}_dimer"
        ovl_dict[dk] = first[:dn, :dn].astype(np.float32, copy=True)
    return ovl_dict, records


def _write_bundle(path, ovl_dict, means_dict, system):
    data = {}
    data.update(ovl_dict)
    data.update(means_dict)
    write_datasets(path, data, attrs={"dimer_n": system.dimer_n},
                   sym_map=system.c2_sym_map())
    print(f"  bundle → {path}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    args = _cli()
    if args.system_preset == "lc8":
        system = lc8_system()
    elif args.chain_length:
        system = DimerSystem(chain_length=args.chain_length)
    else:
        sys.exit("ERROR: --chain-length required for custom systems")

    if args.dimer_n and args.dimer_n != system.dimer_n:
        system = DimerSystem(
            chain_length=args.dimer_n // 2,
            peptide_seqs=system.peptide_seqs,
            binding_windows_1=system.binding_windows_1,
            binding_windows_2=system.binding_windows_2,
        )

    tag1 = parse_tag(args.tag1)
    ovl_dir = Path(args.ovl_dir)

    if args.mode == "intra":
        print(f"[dimer-ovl] intra: {tag1}")
        objs = _build_objects(tag1, args.nrep, args.hist_dir)
        matrix_n = _probe_matrix_n(objs, args.hist_dset)
        ovl_d, recs = _run_intra(tag1, objs, system, args.hist_dset, args.edges_dset)
        bk, unbl, _ = compute_block_aware_mean(recs, system, matrix_n, args.dd_atol)
        md = build_means_dict(bk, unbl)
        bp = ovl_dir / f"ovl_results_{tag1}_vs_{tag1}_all_replicates.h5"
        _write_bundle(bp, ovl_d, md, system)
        print(f"[done] {len(ovl_d)} OVL, {len(bk)} buckets")
        return

    if args.tag2 is None:
        sys.exit("ERROR: tag2 required for --mode pair")
    tag2 = parse_tag(args.tag2)
    print(f"[dimer-ovl] pair: {tag1} vs {tag2}")

    objs1 = _build_objects(tag1, args.nrep, args.hist_dir)
    objs2 = _build_objects(tag2, args.nrep, args.hist_dir)
    matrix_n = _probe_matrix_n(objs1, args.hist_dset)

    cross_ovl, cross_recs = _run_cross(
        tag1, tag2, objs1, objs2, system, args.hist_dset, args.edges_dset)
    intra1_ovl, intra1_recs = _run_intra(
        tag1, objs1, system, args.hist_dset, args.edges_dset)
    intra2_ovl, intra2_recs = _run_intra(
        tag2, objs2, system, args.hist_dset, args.edges_dset)

    cb, cu, _ = compute_block_aware_mean(cross_recs, system, matrix_n, args.dd_atol)
    i1b, i1u, _ = compute_block_aware_mean(intra1_recs, system, matrix_n, args.dd_atol)
    i2b, i2u, _ = compute_block_aware_mean(intra2_recs, system, matrix_n, args.dd_atol)

    cm = build_means_dict(cb, cu)
    i1m = build_means_dict(i1b, i1u)
    i2m = build_means_dict(i2b, i2u)

    write_datasets(ovl_dir / f"ovl_cross_{tag2}_vs_{tag1}.h5",
                   cross_ovl, sym_map=system.c2_sym_map())
    write_datasets(ovl_dir / f"ovl_boundstate_{tag2}_vs_{tag1}.h5",
                   cm, sym_map=system.c2_sym_map())

    _write_bundle(ovl_dir / f"ovl_results_{tag2}_vs_{tag1}_all_replicates.h5",
                  cross_ovl, cm, system)

    if not args.skip_intra_bundles:
        for tag, od, md in [(tag1, intra1_ovl, i1m), (tag2, intra2_ovl, i2m)]:
            _write_bundle(ovl_dir / f"ovl_results_{tag}_vs_{tag}_all_replicates.h5",
                          od, md, system)

    print(f"[done] cross={len(cross_ovl)} i1={len(intra1_ovl)} i2={len(intra2_ovl)}")


def _probe_matrix_n(objs, hist_dset):
    import h5py
    from coop_stat.io.h5 import pick_3d_dataset
    for obj in objs:
        with h5py.File(obj["path"], "r") as h5:
            return h5[pick_3d_dataset(h5, preferred=hist_dset)].shape[0]
    raise RuntimeError("No histogram files found")


if __name__ == "__main__":
    main()
