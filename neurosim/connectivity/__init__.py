# Connectivity module for NeuroSim
"""
Tools for analyzing brain connectivity including Granger causality.
"""

from .granger import granger_causality_matrix, causality_vs_correlation_summary

__all__ = ["granger_causality_matrix", "causality_vs_correlation_summary"]
