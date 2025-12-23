"""
FEDSTR Python ML Module
=======================

This module contains the machine learning training code for FEDSTR.
"""

from .train_tiny import train_model, load_and_evaluate

__all__ = ['train_model', 'load_and_evaluate']
