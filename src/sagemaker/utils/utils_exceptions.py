"""
Custom exceptions for YOLO training and inference.
"""


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class InferenceError(Exception):
    """Exception raised during inference."""
    pass

