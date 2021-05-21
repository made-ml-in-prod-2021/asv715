"""
Dataclass for split data params
"""
from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    """
    Dataclass for split data params
    """
    val_size: float = field(default=0.2)
    random_state: int = field(default=13)
    shuffle: bool = field(default=True)
