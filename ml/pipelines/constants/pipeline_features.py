from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineFeatures:
    input_features: list[str]
    derived_features: list[str]
    categorical_features: list[str]
    selected_features: list[str]