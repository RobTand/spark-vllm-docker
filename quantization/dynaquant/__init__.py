"""DynaQuant: mixed-precision quantization allocator for LLMs.

Model-agnostic. Format-agnostic via the registry in format_registry.py.
Sensitivity measurement via Fisher trace or Hutchinson Hessian diagonal.
Closed-loop: predicted layer loss increase uses MEASURED quantization
error (not hand-tuned analytical constants).

Pipeline:
    1. sensitivity_probe.py       measure per-Linear sensitivity
    2. measure_quant_cost.py      measure per-Linear × per-format ΔMSE (RTN)
    3. mixed_precision_allocator.py  solve assignment, emit AutoRound config
"""
from .format_registry import FormatSpec, REGISTRY, register_format
