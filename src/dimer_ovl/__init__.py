"""
dimer_ovl — OVL-based statistical comparison of symmetric homodimer MD simulations.

Pipeline:
    distance H5s → histograms → OVL matrices → statistical comparison

Core entry point is DimerSystem, which replaces all hardcoded topology constants.

Example
-------
>>> from dimer_ovl import DimerSystem
>>> system = DimerSystem(
...     chain_length=89,
...     peptide_seqs={"BSN2": "YPRATAEFSTQTPSP", "SPAG5": "YHPETQDSSTQTDTS"},
...     binding_windows_1=[(1, 24), (59, 89), (114, 147)],
...     binding_windows_2=[(25, 58), (90, 113), (148, 178)],
... )

A key feature of the binding windows is that the binding groove is at the dimer interface, sharing residues 
from monomer 1 (1-89) and monomer 2 (90-178). 
"""

from dimer_ovl.config import DimerSystem
from dimer_ovl.tags import Tag, parse_tag

__all__ = ["DimerSystem", "Tag", "parse_tag"]
