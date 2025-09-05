#!/usr/bin/env python3
"""
Strategic Coalition SHAP Implementation

This module provides the main StrategicCoalitionSHAP class - a memory-efficient
alternative to exact Kernel SHAP with O(mk) complexity.
"""

# Import the main implementation
from .clean_strategic_coalition_shap import StrategicCoalitionSHAP

# Make it available at module level
__all__ = ['StrategicCoalitionSHAP']
