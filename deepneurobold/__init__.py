"""
DeepNeuroBOLD
=============
BOLD fMRI-based tumor infiltration mapping using supervised machine learning.

Modules
-------
core          : Abstract base classes shared across all modules.
preprocessing : DICOM conversion, BOLD registration, tissue segmentation.
data          : BOLD loading, segmentation masks, train/val sampling.
features      : BOLD feature extraction and representation builders.
classifiers   : Random Forest, Logistic Regression, SVM classifiers.
analysis      : Hotspot detection, clustering, and visualization.
experiment    : Experiment specification, registry, and runner.
"""

__version__ = "0.1.0"
