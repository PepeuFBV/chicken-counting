import torch
from typing import Optional


def load_checkpoint(path: str, model: torch.nn.Module, map_location: Optional[str] = None) -> torch.nn.Module:
    """Load a checkpoint into model. Returns the model.

    If checkpoint contains a `state_dict` key, it will be used. Otherwise the
    whole file is assumed to be a state_dict.
    """
    ck = torch.load(path, map_location=map_location or "cpu")
    if isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
    else:
        state = ck
    # try to load with strict=False to allow minor shape/name differences
    model.load_state_dict(state, strict=False)
    return model


__all__ = ["load_checkpoint"]
