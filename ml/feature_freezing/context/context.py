from dataclasses import dataclass

@dataclass(frozen=True)
class FreezeContext:
    problem: str
    segment: str
    feature_set: str
    version: str
