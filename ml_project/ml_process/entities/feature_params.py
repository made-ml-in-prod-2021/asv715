"""
Dataclass to work with features
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    """
    Dataclass to work with features
    """
    categorical_features: Optional[List[str]]
    numerical_features: List[str]
    target_col: Optional[str]
    use_log_trick: bool = field(default=False)
