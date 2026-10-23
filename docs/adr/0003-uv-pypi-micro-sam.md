# 0003: uv + PyPI micro-sam instead of conda-forge

Status: accepted (2026-10-12)

## Context

The prototype managed its environment with conda (micro-sam from
conda-forge), which pinned the workflow to machines with conda set up
correctly and gave no reproducible environment for a new machine or a new
user. The package must be installable with one command.

## Decision

The project is managed with **uv**: hatchling build backend, Python ≥ 3.12
with the interpreter pinned to 3.13, and a committed `uv.lock`.
**micro-sam comes from PyPI** as a hard dependency alongside numpy, scipy,
scikit-image, tifffile, Pillow, matplotlib, torch_em, h5py, and tomli-w.
There is no conda anywhere in the workflow. micro-sam is imported lazily
inside the segment step only, so `difference`, `analyze`, and `train` do
not pay the torch import cost.

## Consequences

- One command (`uv sync`) reproduces the exact environment, including
  micro-sam, on any machine with uv.
- torch comes from PyPI rather than conda-forge and may differ from the
  prototype's conda torch, so small pixel-level segmentation differences
  against the prototype's outputs are expected. They are covered by visual
  QA and number sanity in the one-time full-plate check rather than byte
  identity; that trade-off is an explicit user decision.
- The finetuning track (micro-sam `train_sam`) runs in the same uv
  environment.
