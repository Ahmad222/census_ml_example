"""
Utility functions for the Census Income Prediction project.
"""

import os
import pickle
import joblib
from pathlib import Path
from typing import Any, Optional


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filepath: str) -> None:
    """Save a trained model to disk."""
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load a trained model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def save_pickle(obj: Any, filepath: str) -> None:
    """Save an object as pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load an object from pickle file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filepath}")
    return obj

