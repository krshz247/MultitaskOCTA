"""Compatibility shim for the legacy scripts: the pipeline code moved to the octa package."""
from octa.config import create_train_arg_parser  # noqa: F401
from octa.data.loaders import generate_dataset  # noqa: F401
