"""Compatibility shim for the legacy scripts: the network code moved to octa/models.py."""
from octa.models import *  # noqa: F401,F403
from octa.models import my_get_encoder, MyMultibranchModel, MyUnetDecoder  # noqa: F401
