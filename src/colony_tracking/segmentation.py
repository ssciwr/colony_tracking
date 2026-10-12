"""Segmentation stage: growth-mask-gated instance segmentation (ADR 0002).

The growth mask is built from the final frame and gates detection in earlier
frames, including the drift gate. The model-based part uses micro-sam,
imported lazily so that difference/analyze/train do not pay the torch import
cost. Implemented in a later ticket.
"""
