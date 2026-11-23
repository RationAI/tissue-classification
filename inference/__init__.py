"""Standalone inference for running the tissue classifier outside the MUNI cluster.

No Ray, MLflow or kube_jobs: one process, one GPU, slides read directly from disk.
See ``inference/README.md``.
"""
