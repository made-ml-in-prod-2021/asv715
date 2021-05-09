from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: Optional[List[str]]
    numerical_features: List[str]
    #binary_features: List[str]
    #features_to_drop: List[str]
    target_col: Optional[str]
    use_log_trick: bool = field(default=False)
