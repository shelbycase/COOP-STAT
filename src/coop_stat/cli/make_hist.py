"""CLI: ``dimer-hist`` — build per-pair distance histograms from distance H5 files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from coop_stat.config import DimerSystem, lc8_system
from coop_stat.histogram.builder import (
    build_edges,
    build_histogram,
    compute_pairwise_minmax,
    write_edges_h5,
    write_histogram_h5,
)
from coop_stat.histogram.indexing import canonical_size
from coop_stat.histogram.sanity import post_check_histogram, pre_check_resids
from coop_stat.tags import parse_tag, Tag


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build distance histograms for OVL pipeline.")
    p.add_argument("--tags", required=True, help="Comma-separated system tags")
    p.add_argument("--reps", default="1-3", help="Replicates: '1-3' or '1,2,3'")
    p.add_argument("--dist-dir", default="distance_h5")
    p.add_argument("--out-dir", default="hist_h5")
    p.add_argument("--edges-h5", required=True, help="Canonical bin edges H5 (read or write)")
    p.add_argument("--make-edges", action="store_true", help="Compute edges (pass 1)")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--start-file", type=int, default=1667,
                   help="Integer in dist H5 filename (e.g. 1667)")
    p.add_argument("--dist-dset", default="distances")
    p.add_argument("--counts-dset", default="hist_counts")
    p.add_argument("--edges-dset", default="bin_edges")
    p.add_argument("--resid-dset", default="resids")
    p.add_argument("--sanity-n", type=int, default=20,
                   help="Post-check pairs per histogram (0 to disable)")
    p.add_argument("--sanity-tol", type=float, default=1.5)
    p.add_argument("--system-preset", default="lc8",
                   choices=["lc8", "custom"],
                   help="Use a built-in system config")
    p.add_argument("--chain-length", type=int, default=None,
                   help="Chain length for custom systems")
    return p.parse_args()


def _parse_reps(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = map(int, spec.split("-"))
        return list(range(a, b + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _dist_path(dist_dir: str, tag: Tag, rep: int, start_file: int) -> Path:
    return Path(dist_dir) / f"{tag}_{rep}_start{start_file}_dist.h5"


def main():
    args = _cli()

    # Build system config
    if args.system_preset == "lc8":
        system = lc8_system()
    elif args.chain_length:
        system = DimerSystem(chain_length=args.chain_length)
    else:
        sys.exit("ERROR: --chain-length required for custom systems")

    tags = [parse_tag(t.strip()) for t in args.tags.split(",") if t.strip()]
    reps = _parse_reps(args.reps)

    # Determine canonical size
    import h5py
    from coop_stat.io.h5 import pick_3d_dataset
    sizes = set()
    for tag in tags:
        for r in reps:
            p = _dist_path(args.dist_dir, tag, r, args.start_file)
            if p.exists():
                with h5py.File(p, "r") as h5:
                    raw_n = h5[pick_3d_dataset(h5, args.dist_dset)].shape[0]
                sizes.add(canonical_size(tag, raw_n))
                break
    if len(sizes) != 1:
        sys.exit(f"ERROR: inconsistent canonical sizes: {sizes}")
    n_canon = next(iter(sizes))
    print(f"[canon] size = {n_canon}")

    # Build tagged_paths
    tagged_paths = []
    for tag in tags:
        for r in reps:
            p = _dist_path(args.dist_dir, tag, r, args.start_file)
            for view in tag.histogram_views:
                tagged_paths.append((tag, view, p))

    if args.make_edges:
        print("[mode] computing shared canonical bin edges")
        dmin, dmax = compute_pairwise_minmax(
            tagged_paths, n_canon,
            dist_dset=args.dist_dset,
            start_frame=args.start_frame,
            resid_dset=args.resid_dset,
        )
        edges = build_edges(dmin, dmax, args.bins)
        write_edges_h5(
            Path(args.edges_h5), edges,
            edges_dset=args.edges_dset,
            n_bins=args.bins,
            tags=[str(t) for t in tags],
        )
        print(f"[done] edges → {args.edges_h5}")
        return

    # Pass 2: histograms
    import h5py
    with h5py.File(args.edges_h5, "r") as h5:
        canon_edges = h5[args.edges_dset][...]

    for tag in tags:
        for r in reps:
            dist_h5 = _dist_path(args.dist_dir, tag, r, args.start_file)
            for view in tag.histogram_views:
                out = Path(args.out_dir) / str(tag) / (
                    f"{tag}_{r}_diststart{args.start_file}_hist{args.bins}bins.h5"
                    if view == "native"
                    else f"{tag}_{r}_{view}_diststart{args.start_file}_hist{args.bins}bins.h5"
                )
                print(f"[hist] {tag} rep={r} view={view}")
                counts, edges, canon_idx, canon_resids = build_histogram(
                    tag, view, dist_h5, canon_edges,
                    start_frame=args.start_frame,
                    dist_dset=args.dist_dset,
                    resid_dset=args.resid_dset,
                )
                write_histogram_h5(
                    out, counts, edges, tag, r, view,
                    canon_resids=canon_resids,
                    counts_dset=args.counts_dset,
                    edges_dset=args.edges_dset,
                )
                if args.sanity_n > 0:
                    result = post_check_histogram(
                        out, dist_h5, canon_idx,
                        dimer_n=system.dimer_n,
                        n_check=args.sanity_n,
                        tol=args.sanity_tol,
                        counts_dset=args.counts_dset,
                        edges_dset=args.edges_dset,
                        dist_dset=args.dist_dset,
                        start_frame=args.start_frame,
                    )
                    status = result["status"]
                    print(f"  [sanity] {status}")
                    if status == "FAIL":
                        sys.exit(f"ERROR: sanity check failed for {tag} rep={r} {view}")

    print("[done] all histograms written.")


if __name__ == "__main__":
    main()
