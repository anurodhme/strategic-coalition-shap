#!/usr/bin/env python3
"""
Low-Rank SHAP Implementation

This module provides the main LowRankSHAP class - a memory-efficient
alternative to exact Kernel SHAP with O(nk) complexity.
"""

# Import the main implementation
from .clean_lowrank_shap import LowRankSHAP

# Make it available at module level
__all__ = ['LowRankSHAP']
