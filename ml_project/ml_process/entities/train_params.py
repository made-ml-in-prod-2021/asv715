from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="LogisticRegression")
    random_state: int = field(default=255)
    random_forest_estimators: int = field(default=100)
    max_depth: int = field(default=None)
    min_samples_split: int = field(default=3)
    min_samples_leaf: int = field(default=1)
    log_reg_max_iterations: int = field(default=100)
