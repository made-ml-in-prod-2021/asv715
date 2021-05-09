from dataclasses import dataclass, field


@dataclass()
class ProcessingParams:
    use_scaler: bool = field(default=False)
    use_poly_features: bool = field(default=False)
    poly_features_max_power: int = field(default=1)
